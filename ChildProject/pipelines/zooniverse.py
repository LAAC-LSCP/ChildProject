import argparse
import datetime
import itertools
import json
import math
import multiprocessing as mp
import os
import pandas as pd
from panoptes_client import Panoptes, Project, Subject, SubjectSet, Classification
import shutil
import subprocess
import sys

from pydub import AudioSegment

import ChildProject
from ChildProject.pipelines.pipeline import Pipeline

def pad_interval(onset: int, offset: int, chunks_length: int, chunks_min_amount: int = 1):
    length = offset-onset

    target_length = chunks_length * max(chunks_min_amount, math.ceil(length/chunks_length))
    onset -= (target_length - length)/2
    offset += (target_length - length)/2

    return int(onset), int(offset)

class Chunk():
    def __init__(self, recording, onset, offset, segment_onset, segment_offset):
        self.recording = recording
        self.onset = onset
        self.offset = offset

        self.segment_onset = segment_onset
        self.segment_offset = segment_offset


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
                
    def split_recording(self, segments: pd.DataFrame) -> list:
        segments = segments.to_dict(orient = 'records')
        chunks = []

        source = os.path.join(self.project.path, ChildProject.projects.ChildProject.RAW_RECORDINGS, segments[0]['recording_filename'])
        audio = AudioSegment.from_wav(source)

        print("extracting chunks from {}...".format(source))

        for segment in segments:
            onset = int(segment['segment_onset']*1000)
            offset = int(segment['segment_offset']*1000)

            original_onset = onset
            original_offset = offset

            if self.chunks_length > 0:
                onset, offset = pad_interval(onset, offset, self.chunks_length, self.chunks_min_amount)

                if onset < 0:
                    print("skipping chunk with negative onset ({})".format(onset))
                    continue

                intervals = [(a, a+self.chunks_length) for a in range(onset, offset, self.chunks_length)]
            else:
                intervals = [(onset, offset)]

            for (onset, offset) in intervals:
                chunk = Chunk(
                    segment['recording_filename'],
                    onset, offset,
                    original_onset, original_offset
                )
                chunk_audio = audio[chunk.onset:chunk.offset].fade_in(10).fade_out(10)

                wav = os.path.join(self.destination, 'chunks', chunk.getbasename('wav'))
                mp3 = os.path.join(self.destination, 'chunks', chunk.getbasename('mp3'))

                if not os.path.exists(wav):
                    chunk_audio.export(wav, format = 'wav')

                if not os.path.exists(mp3):
                    chunk_audio.export(mp3, format = 'mp3')

                chunks.append(chunk)

        return chunks

    def extract_chunks(self, path: str, destination: str, keyword: str, 
        segments: str,
        batch_size: int = 1000,
        chunks_length: int = -1, chunks_min_amount: int = 1,
        exclude_segments: list = [],
        threads: int = 0,
        **kwargs):
        """extract-audio chunks based on a list of segments and prepare them for upload
        to zooniverse.

        :param path: dataset path
        :type path: str
        :param destination: path to the folder where to store the metadata and audio chunks
        :type destination: str
        :param segments: path to the input segments csv dataframe, defaults to None
        :type segments: str
        :param keyword: keyword to insert in the output metadata
        :type keyword: str
        :param batch_size: size of each zooniverse subject subset, defaults to 1000
        :type batch_size: int, optional
        :param chunks_length: length of the chunks, in milliseconds, defaults to -1
        :type chunks_length: int, optional
        :param chunks_min_amount: minimum amount of chunk per segment, defaults to 1
        :type chunks_min_amount: int, optional
        :param threads: amount of threads to run-on, defaults to 0
        :type threads: int, optional
        :param exclude_segments: unused, defaults to []
        :type exclude_segments: list, optional
        """


        parameters = locals()
        parameters = [[key, parameters[key]] for key in parameters if key != 'self']

        self.destination = destination
        self.project = ChildProject.projects.ChildProject(path)

        batch_size = int(batch_size)
        self.chunks_length = int(chunks_length)
        self.chunks_min_amount = chunks_min_amount
        threads = int(threads)

        destination_path = os.path.join(destination, 'chunks')
        os.makedirs(destination_path, exist_ok = True)
        if os.listdir(destination_path):
            raise ValueError("destination '{}' is not empty, please choose another destination.".format(destination_path))

        self.segments = pd.read_csv(segments)
        shutil.copyfile(segments, os.path.join(self.destination, 'segments.csv'))

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
            'segment_onset': c.segment_onset,
            'segment_offset': c.segment_offset,
            'wav': c.getbasename('wav'),
            'mp3': c.getbasename('mp3'),
            'date_extracted': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'uploaded': False,
            'project_id': '',
            'subject_set': '',
            'zooniverse_id': 0,
            'keyword': keyword
        } for c in self.chunks])

        # shuffle chunks so that they can't be joined back together
        # based on Zooniverse subject IDs
        self.chunks = self.chunks.sample(frac=1).reset_index(drop=True)
        self.chunks['batch'] = self.chunks.index.map(lambda x: int(x/batch_size))
        self.chunks.index.name = 'index'
        self.chunks.to_csv(os.path.join(self.destination, 'chunks.csv'))

        parameters.extend([
            ['version', ChildProject.__version__],
            ['date_extracted', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
        ])

        pd.DataFrame(
            data = parameters,
            columns = ['param', 'value']
        ).to_csv(os.path.join(self.destination, 'parameters.csv'), index = False)

    def upload_chunks(self, destination, project_id, set_prefix, zooniverse_login, zooniverse_pwd, batches = 0, **kwargs):
        self.destination = destination 

        metadata_location = os.path.join(self.destination, 'chunks.csv')
        try:
            self.chunks = pd.read_csv(metadata_location, index_col = 'index')
        except:
            raise Exception("cannot read chunk metadata in {}. Check the --destination parameter, and make sure you have extracted chunks before.".format(metadata_location))

        Panoptes.connect(username = zooniverse_login, password = zooniverse_pwd)
        zooniverse_project = Project(project_id)

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
                chunk['project_id'] = project_id
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

        classifications = pd.DataFrame(classifications)
        classifications['user_id'] = classifications['links'].apply(lambda s: s['user'])
        classifications['subject_id'] = classifications['links'].apply(lambda s: s['subjects'][0])
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

        parser_extraction = subparsers.add_parser('extract-chunks', help = 'extract chunks to DESTINATION, proving all associate metadata in the same directory')
        parser_extraction.add_argument('path', help = 'path to the dataset')
        parser_extraction.add_argument('--keyword', help = 'export keyword', required = True)

        parser_extraction.add_argument('--chunks-length', help = 'chunk length (in milliseconds). if <= 0, the segments will not be split into chunks', type = int, default = 0)
        parser_extraction.add_argument('--chunks-min-amount', help = 'minimum amount of chunks to extract from a segment', default = 1)

        parser_extraction.add_argument('--segments', help = 'path to the input segments dataframe', required = True)
        parser_extraction.add_argument('--destination', help = 'destination', required = True)
        parser_extraction.add_argument('--exclude-segments', help = 'segments to exclude before sampling', nargs = '+', default = [])
        parser_extraction.add_argument('--batch-size', help = 'batch size', default = 1000, type = int)
        parser_extraction.add_argument('--threads', help = 'how many threads to run on', default = 0, type = int)

        parser_upload = subparsers.add_parser('upload-chunks', help = 'upload chunks and updates DESTINATION/chunks.csv')
        parser_upload.add_argument('--destination', help = 'destination', required = True)
        parser_upload.add_argument('--zooniverse-login', help = 'zooniverse login', required = True)
        parser_upload.add_argument('--zooniverse-pwd', help = 'zooniverse password', required = True)
        parser_upload.add_argument('--project-id', help = 'zooniverse project id', required = True)
        parser_upload.add_argument('--set-prefix', help = 'subject prefix', required = True)
        parser_upload.add_argument('--batches', help = 'amount of batches to upload', required = False, type = int, default = 0)

        parser_retrieve = subparsers.add_parser('retrieve-classifications', help = 'retrieve classifications and save them into DESTINATION/classifications.csv')
        parser_retrieve.add_argument('--destination', help = 'destination', required = True)
        parser_retrieve.add_argument('--zooniverse-login', help = 'zooniverse login', required = True)
        parser_retrieve.add_argument('--zooniverse-pwd', help = 'zooniverse password', required = True)
        parser_retrieve.add_argument('--project-id', help = 'zooniverse project id', required = True)