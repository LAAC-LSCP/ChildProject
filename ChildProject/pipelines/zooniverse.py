import argparse
import datetime
import itertools
import json
import multiprocessing as mp
import os
import pandas as pd
from panoptes_client import Panoptes, Project, Subject, SubjectSet, Classification
import subprocess
import sys

from pydub import AudioSegment

from ChildProject.projects import ChildProject
from ChildProject.annotations import AnnotationManager
from ChildProject.pipelines.pipeline import Pipeline

def check_dur(dur, target):
    if dur % 0.5 == 0:
        new_dur=dur
        remain=0
    else:
        closest_int=int(round(dur))
        if closest_int>=dur:
            new_dur = float(closest_int)
        else:
            new_dur = float(closest_int)+0.5
    remain = float(new_dur-dur)
    return new_dur,remain

class Chunk():
    def __init__(self, recording, onset, offset):
        self.recording = recording
        self.onset = onset
        self.offset = offset

    def getbasename(self, extension):
        return "{}_{}_{}.{}".format(
            os.path.splitext(self.recording.replace('/', '_'))[0],
            self.onset,
            self.offset,
            extension
        )

class ZooniversePipeline(Pipeline):
    def __init__(self):
        self.chunks = []
                
    def split_recording(self, segments):
        segments = segments.sample(self.sample_size).to_dict(orient = 'records')

        source = os.path.join(self.project.path, 'recordings', segments[0]['recording_filename'])
        audio = AudioSegment.from_wav(source)

        for segment in segments:
            onset = int(segment['segment_onset']*1000)
            offset = int(segment['segment_offset']*1000)
            difference = offset-onset

            if difference < 1000:
                tgt = 1000-difference
                onset = float(onset)-tgt/2
                offset = float(offset) + tgt/2
            else:
                new_dur,remain = check_dur((offset-onset)/1000, self.chunk_length/1000)
                onset = float(onset)-remain*1000/2
                offset = float(offset) + remain*1000/2

            onset = int(onset)
            offset = int(offset)

            intervals = range(onset, offset, self.chunk_length) 
            chunks = []

            for interval in intervals:
                chunk = Chunk(segment['recording_filename'], interval, interval + self.chunk_length)
                chunk_audio = audio[chunk.onset:chunk.offset].fade_in(10).fade_out(10)

                wav = os.path.join(self.destination, 'chunks', chunk.getbasename('wav'))
                mp3 = os.path.join(self.destination, 'chunks', chunk.getbasename('mp3'))

                if not os.path.exists(wav):
                    chunk_audio.export(wav, format = 'wav')

                if not os.path.exists(mp3):
                    chunk_audio.export(mp3, format = 'mp3')

                chunks.append(chunk)

            return chunks

    def extract_chunks(self, destination, path, annotation_set = 'vtc',
        batch_size = 1000, target_speaker_type = 'CHI', sample_size = 500, chunk_length = 500, threads = 0, batches = 0, **kwargs):

        assert 1000 % chunk_length == 0, 'chunk_length should divide 1000'


        self.destination = destination
        self.project = ChildProject(path)

        batch_size = int(batch_size)
        sample_size = int(sample_size)
        chunk_length = int(chunk_length)
        threads = int(threads)

        self.sample_size = sample_size
        self.chunk_length = chunk_length

        am = AnnotationManager(self.project)
        self.annotations = am.annotations
        self.annotations = self.annotations[self.annotations['set'] == annotation_set]
        self.segments = am.get_segments(self.annotations)
        self.segments = self.segments[self.segments['speaker_type'] == target_speaker_type]
        self.segments['segment_onset'] = self.segments['segment_onset'] + self.segments['time_seek']
        self.segments['segment_offset'] = self.segments['segment_offset'] + self.segments['time_seek']

        destination_path = os.path.join(destination, 'chunks')
        os.makedirs(destination_path, exist_ok = True)
        if os.listdir(destination_path):
            raise ValueError("destination '{}' is not empty, please choose another destination.".format(destination_path))

        segments = []
        for _recording, _segments in self.segments.groupby('recording_filename'):
            segments.append(_segments.assign(recording_filename = _recording))
        
        pool = mp.Pool(threads if threads > 0 else mp.cpu_count())
        self.chunks = pool.map(self.split_recording, segments)
        self.chunks = itertools.chain.from_iterable(self.chunks)
        self.chunks = pd.DataFrame([{
            'recording': c.recording,
            'onset': c.onset,
            'offset': c.offset,
            'wav': c.getbasename('wav'),
            'mp3': c.getbasename('mp3'),
            'speaker_type': target_speaker_type,
            'date_extracted': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'uploaded': False,
            'project_slug': '',
            'subject_set': '',
            'zooniverse_id': 0
        } for c in self.chunks])

        # shuffle chunks so that they can't be joined back together
        # based on Zooniverse subject IDs
        self.chunks = self.chunks.sample(frac=1).reset_index(drop=True)
        self.chunks['batch'] = self.chunks.index.map(lambda x: int(x/batch_size))
        self.chunks.index.name = 'index'
        self.chunks.to_csv(os.path.join(self.destination, 'chunks.csv'))

    def upload_chunks(self, destination, project_slug, set_prefix, zooniverse_login, zooniverse_pwd, batches = 0, **kwargs):
        self.destination = destination 

        metadata_location = os.path.join(self.destination, 'chunks.csv')
        try:
            self.chunks = pd.read_csv(metadata_location, index_col = 'index')
        except:
            raise Exception("cannot read chunk metadata in {}. Check the --destination parameter, and make sure you have extracted chunks before.".format(metadata_location))

        Panoptes.connect(username = zooniverse_login, password = zooniverse_pwd)
        zooniverse_project = Project.find(slug = project_slug)

        subjects_metadata = []
        uploaded = 0
        for batch, chunks in self.chunks.groupby('batch'):
            if chunks['uploaded'].all():
                continue

            subject_set = SubjectSet()
            subject_set.links.project = zooniverse_project
            subject_set.display_name = "{}_batch_{}".format(set_prefix, batch)
            subject_set.save()
            subjects = []

            _chunks = chunks.to_dict(orient = 'index')
            for chunk_index in _chunks:
                chunk = _chunks[chunk_index]

                print("uploading chunk {} ({},{}) in batch {}".format(chunk['recording'], chunk['onset'], chunk['offset'], batch))

                subject = Subject()
                subject.links.project = zooniverse_project
                subject.add_location(os.path.join(self.destination, 'chunks', chunk['mp3']))
                subject.metadata['date_extracted'] = chunk['date_extracted']
                subject.save()
                subjects.append(subject)

                chunk['index'] = chunk_index
                chunk['zooniverse_id'] = subject.id
                chunk['project_slug'] = project_slug
                chunk['subject_set'] = str(subject_set.display_name)
                chunk['uploaded'] = True
                subjects_metadata.append(chunk)
            
            subject_set.add(subjects)

            self.chunks.update(
                pd.DataFrame(subjects_metadata).set_index('index')
            )

            self.chunks.to_csv(os.path.join(self.destination, 'chunks.csv'))
            uploaded += 1

            if batches > 0 and uploaded >= batches:
                return

    def retrieve_classifications(self, destination, project_id, zooniverse_login, zooniverse_pwd, **kwargs):
        Panoptes.connect(username = zooniverse_login, password = zooniverse_pwd)
        project = Project(project_id)

        answers_translation_table = []
        for workflow in project.links.workflows:
            workflow_id = workflow.id
            for task_id in workflow.tasks:
                n = 0
                for answer in workflow.tasks[task_id]['answers']:
                    answers_translation_table.append({
                        'workflow_id': str(workflow_id),
                        'task_id': str(task_id),
                        'answer_id': str(n),
                        'answer': answer['label']
                    })
                    n += 1

        answers_translation_table = pd.DataFrame(answers_translation_table)

        classifications = []
        for c in Classification.where(
            scope = 'project',
            page_size = 1000,
            project_id = project_id
        ):
            classifications.append(c.raw)
            if len(classifications) > 2000:
                break

        classifications = pd.DataFrame(classifications)
        classifications['subject_id'] = classifications['links'].apply(lambda s: s['user'])
        classifications['user_id'] = classifications['links'].apply(lambda s: s['subjects'][0])
        classifications['workflow_id'] = classifications['links'].apply(lambda s: s['workflow'])
        classifications['task_id'] = classifications['annotations'].apply(lambda s: str(s[0]['task']))
        classifications['answer_id'] = classifications['annotations'].apply(lambda s: str(s[0]['value']))

        classifications = classifications[['id', 'user_id', 'subject_id', 'task_id', 'answer_id', 'workflow_id']]
        classifications = classifications.merge(
            answers_translation_table,
            left_on = ['workflow_id', 'task_id', 'answer_id'],
            right_on = ['workflow_id', 'task_id', 'answer_id']
        )
        classifications.set_index('id').to_csv(os.path.join(destination, 'classifications.csv'))

    def run(self, action, **kwargs):
        if action == 'extract-chunks':
            self.extract_chunks(**kwargs)
        elif action == 'upload-chunks':
            self.upload_chunks(**kwargs)
        elif action == 'retrieve-classifications':
            self.retrieve_classifications(**kwargs)

    @staticmethod
    def setup_parser(parser):
        subparsers = parser.add_subparsers(help = 'action', dest = 'action')

        parser_extraction = subparsers.add_parser('extract-chunks', help = 'extract chunks and store metadata in DESTINATION/chunks.csv')
        parser_extraction.add_argument('path', help = 'path to the dataset')
        parser_extraction.add_argument('--destination', help = 'destination', required = True)
        parser_extraction.add_argument('--sample-size', help = 'how many samples per recording', required = True, type = int)
        parser_extraction.add_argument('--annotation-set', help = 'annotation set', default = 'vtc')
        parser_extraction.add_argument('--target-speaker-type', help = 'speaker type to get chunks from', default = 'CHI', choices=['CHI', 'OCH', 'FEM', 'MAL'])
        parser_extraction.add_argument('--batch-size', help = 'batch size', default = 1000, type = int)
        parser_extraction.add_argument('--threads', help = 'how many threads to run on', default = 0, type = int)

        parser_upload = subparsers.add_parser('upload-chunks', help = 'upload chunks and updates DESTINATION/chunks.csv')
        parser_upload.add_argument('--destination', help = 'destination', required = True)
        parser_upload.add_argument('--zooniverse-login', help = 'zooniverse login', required = True)
        parser_upload.add_argument('--zooniverse-pwd', help = 'zooniverse password', required = True)
        parser_upload.add_argument('--project-slug', help = 'zooniverse project name', required = True)
        parser_upload.add_argument('--set-prefix', help = 'subject prefix', required = True)
        parser_upload.add_argument('--batches', help = 'amount of batches to upload', required = False, type = int, default = 0)

        parser_retrieve = subparsers.add_parser('retrieve-classifications', help = 'retrieve classifications and save them into DESTINATION/classifications.csv')
        parser_retrieve.add_argument('--destination', help = 'destination', required = True)
        parser_retrieve.add_argument('--zooniverse-login', help = 'zooniverse login', required = True)
        parser_retrieve.add_argument('--zooniverse-pwd', help = 'zooniverse password', required = True)
        parser_retrieve.add_argument('--project-id', help = 'zooniverse project id', required = True)