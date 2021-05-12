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
import traceback
from yaml import dump

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
    def __init__(self, recording_filename, onset, offset, segment_onset, segment_offset):
        self.recording_filename = recording_filename
        self.onset = onset
        self.offset = offset

        self.segment_onset = segment_onset
        self.segment_offset = segment_offset


    def getbasename(self, extension):
        return "{}_{}_{}.{}".format(
            os.path.splitext(self.recording_filename.replace('/', '_'))[0],
            self.onset,
            self.offset,
            extension
        )

class ZooniversePipeline(Pipeline):
    def __init__(self):
        self.chunks = []

    def get_credentials(self, login: str = '', pwd: str = ''):
        """returns input credentials if provided or attempts to read them
        from the environment variables.

        :param login: input login, defaults to ''
        :type login: str, optional
        :param pwd: input password, defaults to ''
        :type pwd: str, optional
        :return: (login, pwd)
        :rtype: (str, str)
        """
        if login and pwd:
            self.zooniverse_login = login
            self.zooniverse_pwd = pwd
            return (self.zooniverse_login, self.zooniverse_pwd)

        if os.getenv('ZOONIVERSE_LOGIN'):
            self.zooniverse_login = os.getenv('ZOONIVERSE_LOGIN')
        else:
            raise Exception("no login specified, and no 'ZOONIVERSE_LOGIN' environment variable found")

        if os.getenv('ZOONIVERSE_PWD'):
            self.zooniverse_pwd = os.getenv('ZOONIVERSE_PWD')
        else:
            raise Exception("no password specified, and no 'ZOONIVERSE_PWD' environment variable found")

        return (self.zooniverse_login, self.zooniverse_pwd)

                
    def _split_recording(self, segments: pd.DataFrame) -> list:
        segments = segments.to_dict(orient = 'records')
        chunks = []

        recording = segments[0]['recording_filename']
        source = self.project.get_recording_path(recording, self.profile)

        audio = AudioSegment.from_file(source)

        print("extracting chunks from {}...".format(source))

        for segment in segments:
            original_onset = int(segment['segment_onset'])
            original_offset = int(segment['segment_offset'])
            onset = original_onset
            offset = original_offset

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
                else:
                    print('{} already exists, exportation skipped.'.format(wav))

                if not os.path.exists(mp3):
                    chunk_audio.export(mp3, format = 'mp3')
                else:
                    print('{} already exists, exportation skipped.'.format(mp3))


                chunks.append(chunk)

        return chunks

    def extract_chunks(self, path: str, destination: str, keyword: str, segments: str,
        chunks_length: int = -1, chunks_min_amount: int = 1,
        profile: str = '',
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
        :param chunks_length: length of the chunks, in milliseconds, defaults to -1
        :type chunks_length: int, optional
        :param chunks_min_amount: minimum amount of chunk per segment, defaults to 1
        :type chunks_min_amount: int, optional
        :param profile: recording profile to extract from. If undefined, raw recordings will be used.
        :type profile: str
        :param threads: amount of threads to run-on, defaults to 0
        :type threads: int, optional
        """


        parameters = locals()
        parameters = [{key: parameters[key]} for key in parameters if key not in ['self', 'kwargs']]
        parameters.extend([{key: kwargs[key]} for key in kwargs])

        self.destination = destination
        self.project = ChildProject.projects.ChildProject(path)

        self.chunks_length = int(chunks_length)
        self.chunks_min_amount = chunks_min_amount
        self.profile = profile

        threads = int(threads)

        destination_path = os.path.join(destination, 'chunks')
        os.makedirs(destination_path, exist_ok = True)

        self.segments = pd.read_csv(segments)
        shutil.copyfile(segments, os.path.join(self.destination, 'segments.csv'))

        segments = []
        for _recording, _segments in self.segments.groupby('recording_filename'):
            segments.append(_segments.assign(recording_filename = _recording))
        
        pool = mp.Pool(threads if threads > 0 else mp.cpu_count())
        self.chunks = pool.map(self._split_recording, segments)
        self.chunks = itertools.chain.from_iterable(self.chunks)
        self.chunks = pd.DataFrame([{
            'recording_filename': c.recording_filename,
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
        self.chunks.index.name = 'index'

        date = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        chunks_path = os.path.join(destination, 'chunks_{}.csv'.format(date))
        parameters_path = os.path.join(destination, 'parameters_{}.yml'.format(date))

        self.chunks.to_csv(chunks_path)
        print("exported chunks metadata to {}".format(chunks_path))
        dump({
            'parameters': parameters,
            'package_version': ChildProject.__version__,
            'date': date
        }, open(parameters_path, 'w+'))
        print("exported extract-chunks parameters to {}".format(parameters_path))

        return chunks_path, parameters_path

    def upload_chunks(self, chunks: str, project_id: int, set_name: str,
        zooniverse_login = '', zooniverse_pwd = '',
        amount: int = 1000,
        **kwargs):
        """Uploads ``amount`` audio chunks from the CSV dataframe `chunks` to a zooniverse project.

        :param chunks: path to the chunk CSV dataframe
        :type chunks: [type]
        :param project_id: zooniverse project id
        :type project_id: int
        :param set_name: name of the subject set
        :type set_name: str
        :param zooniverse_login: zooniverse login. If not specified, the program attempts to get it from the environment variable ``ZOONIVERSE_LOGIN`` instead, defaults to ''
        :type zooniverse_login: str, optional
        :param zooniverse_pwd: zooniverse password. If not specified, the program attempts to get it from the environment variable ``ZOONIVERSE_PWD`` instead, defaults to ''
        :type zooniverse_pwd: str, optional
        :param amount: amount of chunks to upload, defaults to 0
        :type amount: int, optional
        """

        self.chunks_file = chunks
        self.get_credentials(zooniverse_login, zooniverse_pwd)

        metadata_location = os.path.join(self.chunks_file)
        try:
            self.chunks = pd.read_csv(metadata_location, index_col = 'index')
        except:
            raise Exception("cannot read chunk metadata from {}.".format(metadata_location))

        Panoptes.connect(username = self.zooniverse_login, password = self.zooniverse_pwd)
        zooniverse_project = Project(project_id)

        subjects_metadata = []
        uploaded = 0

        subject_set = None

        for ss in zooniverse_project.links.subject_sets:
            if ss.display_name == set_name:
                subject_set = ss

        if subject_set is None:
            subject_set = SubjectSet()
            subject_set.links.project = zooniverse_project
            subject_set.display_name = set_name
            subject_set.save()
        
        subjects = []

        chunks_to_upload = self.chunks[self.chunks['uploaded'] == False].head(amount)
        chunks_to_upload = chunks_to_upload.to_dict(orient = 'index')

        if len(chunks_to_upload) == 0:
            print('nothing left to upload.')
            return

        for chunk_index in chunks_to_upload:
            chunk = chunks_to_upload[chunk_index]

            print("uploading chunk {} ({},{})".format(chunk['recording_filename'], chunk['onset'], chunk['offset']))

            subject = Subject()
            subject.links.project = zooniverse_project
            subject.add_location(os.path.join(os.path.dirname(self.chunks_file), 'chunks', chunk['mp3']))
            subject.metadata['date_extracted'] = chunk['date_extracted']

            try:
                subject.save()
            except Exception as e:
                print("failed to save chunk {}. an exception has occured:\n{}".format(chunk_index, str(e)))
                print(traceback.format_exc())
                print("subject upload halting here.")
                break
            
            subjects.append(subject)

            chunk['index'] = chunk_index
            chunk['zooniverse_id'] = str(subject.id)
            chunk['project_id'] = str(project_id)
            chunk['subject_set'] = str(subject_set.display_name)
            chunk['uploaded'] = True
            subjects_metadata.append(chunk)
            
        subject_set.add(subjects)

        self.chunks.update(
            pd.DataFrame(subjects_metadata).set_index('index')
        )

        self.chunks.to_csv(self.chunks_file)

    def retrieve_classifications(self, destination: str, project_id: int,
        zooniverse_login: str = '', zooniverse_pwd: str = '',
        **kwargs):
        """[summary]

        :param destination: output CSV dataframe destination
        :type destination: str
        :param project_id: zooniverse project id
        :type project_id: int
        :param zooniverse_login: zooniverse login. If not specified, the program attempts to get it from the environment variable ``ZOONIVERSE_LOGIN`` instead, defaults to ''
        :type zooniverse_login: str, optional
        :param zooniverse_pwd: zooniverse password. If not specified, the program attempts to get it from the environment variable ``ZOONIVERSE_PWD`` instead, defaults to ''
        :type zooniverse_pwd: str, optional
        """
        self.get_credentials(zooniverse_login, zooniverse_pwd)

        Panoptes.connect(username = self.zooniverse_login, password = self.zooniverse_pwd)
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
        classifications.set_index('id').to_csv(destination)

    def run(self, action, **kwargs):
        if action == 'extract-chunks':
            return self.extract_chunks(**kwargs)
        elif action == 'upload-chunks':
            return self.upload_chunks(**kwargs)
        elif action == 'retrieve-classifications':
            return self.retrieve_classifications(**kwargs)

    @staticmethod
    def setup_parser(parser):
        subparsers = parser.add_subparsers(help = 'action', dest = 'action')

        parser_extraction = subparsers.add_parser('extract-chunks', help = 'extract chunks to <destination>, and exports the metadata inside of this directory')
        parser_extraction.add_argument('path', help = 'path to the dataset')
        parser_extraction.add_argument('--keyword', help = 'export keyword', required = True)
        parser_extraction.add_argument('--chunks-length', help = 'chunk length (in milliseconds). if <= 0, the segments will not be split into chunks (default value: 0)', type = int, default = 0)
        parser_extraction.add_argument('--chunks-min-amount', help = 'minimum amount of chunks to extract from a segment (default value: 1)', default = 1)
        parser_extraction.add_argument('--segments', help = 'path to the input segments dataframe', required = True)
        parser_extraction.add_argument('--destination', help = 'destination', required = True)
        parser_extraction.add_argument('--profile', help = 'Recording profile to extract the audio clips from. If not specified, raw recordings will be used', default = '')
        parser_extraction.add_argument('--threads', help = 'how many threads to run on', default = 0, type = int)

        parser_upload = subparsers.add_parser('upload-chunks', help = 'upload chunks and updates chunk state')
        parser_upload.add_argument('--chunks', help = 'path to the chunk CSV dataframe', required = True)
        parser_upload.add_argument('--project-id', help = 'zooniverse project id', required = True, type = int)
        parser_upload.add_argument('--set-name', help = 'subject set display name', required = True)
        parser_upload.add_argument('--amount', help = 'amount of chunks to upload', required = False, type = int, default = 1000)
        parser_upload.add_argument('--zooniverse-login', help = 'zooniverse login. If not specified, the program attempts to get it from the environment variable ZOONIVERSE_LOGIN instead', default = '')
        parser_upload.add_argument('--zooniverse-pwd', help = 'zooniverse password. If not specified, the program attempts to get it from the environment variable ZOONIVERSE_PWD instead', default = '')

        parser_retrieve = subparsers.add_parser('retrieve-classifications', help = 'retrieve classifications and save them as <destination>')
        parser_retrieve.add_argument('--destination', help = 'output CSV dataframe destination', required = True)
        parser_retrieve.add_argument('--project-id', help = 'zooniverse project id', required = True, type = int)
        parser_retrieve.add_argument('--zooniverse-login', help = 'zooniverse login. If not specified, the program attempts to get it from the environment variable ZOONIVERSE_LOGIN instead', default = '')
        parser_retrieve.add_argument('--zooniverse-pwd', help = 'zooniverse password. If not specified, the program attempts to get it from the environment variable ZOONIVERSE_PWD instead', default = '')
