import argparse
import datetime
import itertools
import json
import math
import multiprocessing as mp
import os
import pandas as pd
import shutil
import subprocess
import sys
import array
import traceback
import signal
import re
import logging

from typing import List
from yaml import dump

from functools import partial

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from numpy import log10

from pydub import AudioSegment
from pydub.utils import get_array_type

from parselmouth import Sound
from parselmouth import SpectralAnalysisWindowShape

from ..projects import ChildProject
from .pipeline import Pipeline
from ..tables import assert_dataframe, assert_columns_presence
from ..utils import retry_func

from ChildProject import __version__

from typing import Tuple

import time

CHUNKS_DTYPES = {
    'recording_filename':  'string',
    'onset': int,
    'offset': int,
    'segment_onset': int,
    'segment_offset': int,
    'wav': 'string',
    'mp3': 'string',
    'date_extracted': 'string',
    'uploaded': 'boolean',
    'project_id': 'Int64',
    'subject_set': 'string',
    'zooniverse_id': 'Int64',
    'keyword': 'string',
    'subject_set_id': 'Int64',
    'dataset' : 'string',
}

# Create a logger for the module (file)
logger_annotations = logging.getLogger(__name__)
# messages are propagated to the higher level logger (ChildProject), used in cmdline.py
logger_annotations.propagate = True

def pad_interval(
    onset: int, offset: int, chunks_length: int, chunks_min_amount: int = 1
) -> Tuple[int, int]:
    length = offset - onset

    target_length = chunks_length * max(
        chunks_min_amount, math.ceil(length / chunks_length)
    )
    onset -= (target_length - length) / 2
    offset += (target_length - length) / 2

    return int(onset), int(offset)


class Chunk:
    def __init__(
        self, recording_filename, onset, offset, segment_onset, segment_offset
    ):
        self.recording_filename = recording_filename
        self.onset = onset
        self.offset = offset

        self.segment_onset = segment_onset
        self.segment_offset = segment_offset

    def getbasename(self, extension):
        return "{}_{}_{}.{}".format(
            os.path.splitext(self.recording_filename.replace("/", "_"))[0],
            self.onset,
            self.offset,
            extension,
        )

class ZooniversePipeline(Pipeline):
    def __init__(self):
        self.chunks = []

    def get_credentials(self, login: str = "", pwd: str = ""):
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

        if os.getenv("ZOONIVERSE_LOGIN"):
            self.zooniverse_login = os.getenv("ZOONIVERSE_LOGIN")
        else:
            raise Exception(
                "no login specified, and no 'ZOONIVERSE_LOGIN' environment variable found"
            )

        if os.getenv("ZOONIVERSE_PWD"):
            self.zooniverse_pwd = os.getenv("ZOONIVERSE_PWD")
        else:
            raise Exception(
                "no password specified, and no 'ZOONIVERSE_PWD' environment variable found"
            )

        return (self.zooniverse_login, self.zooniverse_pwd)

    def _split_recording(self, segments: pd.DataFrame) -> list:
        
        #from raw sound data and sampling rate, build the spectrogram as a matplotlib figure and return it
        def _create_spectrogram(data,sr):
            snd = Sound(data,sampling_frequency=sr)
            # this parameters were chosen to output a spectrogram useful for zooniverse applications (short sounds from babies) we did not feel the need to have flexibility on them
            spectrogram = snd.to_spectrogram(window_length=0.0075,maximum_frequency=8000, time_step= 0.0001 ,frequency_step = 0.1,window_shape= SpectralAnalysisWindowShape.GAUSSIAN)
            
            fig = plt.figure(figsize=(12, 6.75)) #size of the image, we chose 1200x675 pixels for a better display on zooniverse
            gs = fig.add_gridspec(2, hspace=0, height_ratios=[1, 3]) #2 plots (spectrogram 3x bigger than oscillogram)
            axs = gs.subplots(sharex=True)
            
            #scpectrogram plot
            dynamic_range=65
            X, Y = spectrogram.x_grid(), spectrogram.y_grid()
            sg_db = 10 * log10(spectrogram.values)
            axs[1].pcolormesh(X, Y, sg_db, vmin=sg_db.max() - dynamic_range, cmap='Greys')
            axs[1].set_ylim([spectrogram.ymin, spectrogram.ymax])
            axs[1].set_xlabel("time [s]")
            axs[1].set_ylabel("frequency [Hz]")
            axs[1].tick_params( labelright=True)
            axs[1].set_xlim([snd.xmin, snd.xmax])
            
            #oscillogram plot
            axs[0].plot(snd.xs(), snd.values.T, linewidth=0.5)
            axs[0].set_xlim([snd.xmin, snd.xmax])
            axs[0].set_ylabel("amplitude")
            
            #remove overlapping labels
            ticks = axs[0].yaxis.get_major_ticks()
            if len(ticks) : ticks[0].label1.set_visible(False)
            if len(ticks) > 1 : ticks[1].label1.set_visible(False)
            
            fig.tight_layout()
            
            return fig
        
        segments = segments.to_dict(orient="records")
        chunks = []

        recording = segments[0]["recording_filename"]
        source = self.project.get_recording_path(recording, self.profile)

        audio = AudioSegment.from_file(source)

        logger_annotations.info("extracting chunks from %s...", source)


        for segment in segments:
            original_onset = int(segment["segment_onset"])
            original_offset = int(segment["segment_offset"])
            onset = original_onset
            offset = original_offset

            if self.chunks_length > 0:
                onset, offset = pad_interval(
                    onset, offset, self.chunks_length, self.chunks_min_amount
                )

                if onset < 0:
                    logger_annotations.warning("skipping chunk with negative onset (%s)",onset)
                    continue

                intervals = [
                    (a, a + self.chunks_length)
                    for a in range(onset, offset, self.chunks_length)
                ]
            else:
                intervals = [(onset, offset)]

            for (onset, offset) in intervals:
                chunk = Chunk(
                    segment["recording_filename"],
                    onset,
                    offset,
                    original_onset,
                    original_offset,
                )
                chunk_audio = audio[chunk.onset : chunk.offset].fade_in(10).fade_out(10)

                wav = os.path.join(self.destination, "chunks", chunk.getbasename("wav"))
                mp3 = os.path.join(self.destination, "chunks", chunk.getbasename("mp3"))

                if os.path.exists(wav) and os.path.getsize(wav) > 0:
                    logger_annotations.warning("%s already exists, exportation skipped.", wav)
                else:
                    chunk_audio.export(wav, format="wav")

                if os.path.exists(mp3) and os.path.getsize(mp3) > 0:
                    logger_annotations.warning("%s already exists, exportation skipped.", mp3)
                else:
                    chunk_audio.export(mp3, format="mp3")
                    
                if self.spectro:
                    png = os.path.join(self.destination, "chunks", chunk.getbasename("png"))
                    
                    #convert pydub sound data into raw data that the parselmouth library can use
                    bit_depth = chunk_audio.sample_width * 8
                    array_type = get_array_type(bit_depth)
                    
                    sound = array.array(array_type, chunk_audio._data)
                    sr = chunk_audio.frame_rate
                    fig = _create_spectrogram(sound,sr) #create the plot figure
                    
                    if os.path.exists(png) and os.path.getsize(png) > 0:
                        logger_annotations.warning("%s already exists, exportation skipped.", png)
                    else:
                        fig.savefig(png)
                    plt.close(fig)
                        
                chunks.append(chunk)

        return chunks

    def extract_chunks(
        self,
        path: str,
        destination: str,
        keyword: str,
        segments: str,
        chunks_length: int = -1,
        chunks_min_amount: int = 1,
        spectrogram: bool = False,
        profile: str = "",
        threads: int = 1,
        **kwargs
    ):
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
        :param spectrogram: the extraction generates a png spectrogram, defaults to False
        :type spectrogram: bool, optional
        :param profile: recording profile to extract from. If undefined, raw recordings will be used.
        :type profile: str
        :param threads: amount of threads to run-on, defaults to 0
        :type threads: int, optional
        """

        parameters = locals()
        parameters = [
            {key: parameters[key]}
            for key in parameters
            if key not in ["self", "kwargs"]
        ]
        parameters.extend([{key: kwargs[key]} for key in kwargs])

        self.destination = destination
        self.project = ChildProject(path)
        self.project.read()

        self.chunks_length = int(chunks_length)
        self.chunks_min_amount = int(chunks_min_amount)
        self.spectro = spectrogram
        self.profile = profile

        threads = int(threads)

        destination_path = os.path.join(destination, "chunks")
        os.makedirs(destination_path, exist_ok=True)

        self.segments = pd.read_csv(segments)

        assert_dataframe("segments", self.segments, not_empty=True)
        assert_columns_presence(
            "segments",
            self.segments,
            {"recording_filename", "segment_onset", "segment_offset"},
        )

        shutil.copyfile(segments, os.path.join(self.destination, "segments.csv"))

        segments = []
        for _recording, _segments in self.segments.groupby("recording_filename"):
            segments.append(_segments.assign(recording_filename=_recording))

        if threads == 1:
            self.chunks = map(self._split_recording, segments)
        else:
            with mp.Pool(threads if threads > 0 else mp.cpu_count()) as pool:
                self.chunks = pool.map(self._split_recording, segments)

        self.chunks = itertools.chain.from_iterable(self.chunks)
        self.chunks = pd.DataFrame(
            [
                {
                    "recording_filename": c.recording_filename,
                    "onset": c.onset,
                    "offset": c.offset,
                    "segment_onset": c.segment_onset,
                    "segment_offset": c.segment_offset,
                    "wav": c.getbasename("wav"),
                    "mp3": c.getbasename("mp3"),
                    "png": c.getbasename("png") if self.spectro else "NA",
                    "date_extracted": datetime.datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "uploaded": False,
                    "project_id": pd.NA,
                    "subject_set": "",
                    "subject_set_id": pd.NA,
                    "zooniverse_id": pd.NA,
                    "keyword": keyword,
                    "dataset": self.project.experiment
                }
                for c in self.chunks
            ]
        )

        # shuffle chunks so that they can't be joined back together
        # based on Zooniverse subject IDs
        self.chunks = self.chunks.sample(frac=1).reset_index(drop=True)
        self.chunks.index.name = "index"

        date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        chunks_path = os.path.join(destination, "chunks_{}.csv".format(date))
        parameters_path = os.path.join(destination, "parameters_{}.yml".format(date))

        self.chunks.to_csv(chunks_path)
        logger_annotations.info("exported chunks metadata to %s", chunks_path)
        dump(
            {
                "parameters": parameters,
                "package_version": __version__,
                "date": date,
            },
            open(parameters_path, "w+"),
        )
        logger_annotations.info("exported extract-chunks parameters to %s", parameters_path)

        return chunks_path, parameters_path

    def upload_chunks(
        self,
        chunks: str,
        project_id: int,
        set_name: str,
        zooniverse_login="",
        zooniverse_pwd="",
        amount: int = 1000,
        ignore_errors: bool = False,
        record_orphan: bool = False,
        test_endpoint: bool = False,
        **kwargs
    ):
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
        :param amount: amount of chunks to upload, defaults to 1000
        :type amount: int, optional
        :param ignore_errors: carry on with the upload even if a clip fails, the csv will be updated accordingly, single clip errors are ignored but errors that will repeat (e.g. maximum number of subjects uploaded) will still exit
        :type ignore_errors: bool, optional
        :param record_orphan: when true, chunks that are correctly uploaded but not linked to a subject set (orphan) have their line updated with the subject id, project id and Uploaded flag at True, but subject_set empty. link_orphan_subjects can be used to reattempt it. If false, the chunk is considered not uploaded.   
        :type record_orphan: bool, optional
        :param test_endpoint: run this command for tests, operations with zooniverse arefaked and considered succesfull
        :type test_endpoint: bool, optional
        """

        self.chunks_file = chunks
        self.get_credentials(zooniverse_login, zooniverse_pwd)

        metadata_location = os.path.join(self.chunks_file)
        try:
            self.chunks = pd.read_csv(metadata_location, index_col="index", dtype=CHUNKS_DTYPES)
        except:
            raise Exception(
                "cannot read chunk metadata from {}.".format(metadata_location)
            )

        assert_dataframe("chunks", self.chunks)
        assert_columns_presence(
            "chunks",
            self.chunks,
            {"recording_filename", "onset", "offset", "uploaded", "mp3"},
        )
        
        if test_endpoint:
            from .fake_panoptes import Panoptes, Project, Subject, SubjectSet, PanoptesAPIException, reset_tests, TEST_MAX_SUBJECT
            reset_tests()
            if test_endpoint == 2: Subject.max_subjects = TEST_MAX_SUBJECT
        else:
            from panoptes_client import Panoptes, Project, Subject, SubjectSet
            from panoptes_client.panoptes import PanoptesAPIException


        Panoptes.connect(username=self.zooniverse_login, password=self.zooniverse_pwd)
        zooniverse_project = Project(project_id)

        self.subjects_metadata = []
    
        subject_set = None

        for ss in zooniverse_project.links.subject_sets:
            if ss.display_name == set_name:
                subject_set = ss

        if subject_set is None:
            subject_set = SubjectSet()
            subject_set.links.project = zooniverse_project
            subject_set.display_name = set_name
            subject_set.save()

        chunks_to_upload = self.chunks[self.chunks["uploaded"] == False].head(amount)
        chunks_to_upload = chunks_to_upload.to_dict(orient="index")

        if len(chunks_to_upload) == 0:
            logger_annotations.warning("nothing left to upload.")
            return
        
        self.orphan_chunks = []
    
        #handling sigterm and sigint to write to a csv file before exiting to keep track of what was done
        #this is hard to test on a controlled environment, for now, manual testing is required
        original_sigint_handler = signal.getsignal(signal.SIGINT)
        original_sigterm_handler = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, partial(self.exit_upload, rec_orphan=record_orphan, sub_set=set_name))
        signal.signal(signal.SIGTERM, partial(self.exit_upload, rec_orphan=record_orphan, sub_set=set_name))
        
        max_subjects_error = re.compile('User has uploaded \d+ subjects of \d+ maximum')

        for chunk_index in chunks_to_upload:
            chunk = chunks_to_upload[chunk_index]

            logger_annotations.info(
                "uploading chunk %s (%s, %s)",
                chunk["recording_filename"],
                chunk["onset"],
                chunk["offset"], 
                )

            try:
                #we take the mp3 file as the is the format supported by zooniverse
                subject = Subject()
                subject.links.project = zooniverse_project
                subject.add_location(
                    os.path.join(os.path.dirname(self.chunks_file), "chunks", chunk["mp3"])
                )
                subject.metadata["date_extracted"] = chunk["date_extracted"]
                subject.metadata["dataset"] = chunk["dataset"] if 'dataset' in chunk.keys() else ""
                subject.metadata["filename"] = chunk["mp3"]

                subject.save()
            except Exception as e:
                logger_annotations.error(
                "failed to save chunk %s. an exception has occured:\n%s",
                chunk_index,
                str(e),
                )
                logger_annotations.error("%s", traceback.format_exc())

                if ignore_errors and not re.fullmatch(max_subjects_error, str(e)):
                    continue
                else:
                    logger_annotations.error("subject upload halting here.")
                    break
                
            try: 
                retry_func(subject_set.add, PanoptesAPIException, 3, subjects=subject)              
            except PanoptesAPIException as e:
                logger_annotations.error(
                "failed to add subject %s to subject_set %s. An exception has occured:\n%s",
                chunk_index,
                subject_set.display_name,
                str(e),
                )
                logger_annotations.error("%s", traceback.format_exc())
                
                chunk["index"] = chunk_index
                chunk["zooniverse_id"] = subject.id
                chunk["project_id"] = project_id
                chunk["subject_set"] = ""
                chunk['subject_set_id'] = pd.NA
                chunk["uploaded"] = True
                self.orphan_chunks.append(chunk)
                
                if ignore_errors:
                    continue
                else:
                    logger_annotations.error("subject upload halting here.")
                    break

            chunk["index"] = chunk_index
            chunk["zooniverse_id"] = subject.id
            chunk["project_id"] = project_id
            chunk["subject_set"] = str(subject_set.display_name)
            chunk["subject_set_id"] = subject_set.id
            chunk["uploaded"] = True
            self.subjects_metadata.append(chunk)

        if len(self.subjects_metadata) + len(self.orphan_chunks) == 0:
            return

        if len(self.subjects_metadata):
            self.chunks.update(pd.DataFrame(self.subjects_metadata).set_index("index"))

        if record_orphan and len(self.orphan_chunks): 
            self.chunks.update(pd.DataFrame(self.orphan_chunks).set_index("index"))

            logger_annotations.warning(
                "%d chunks were uploaded but not linked to the subject set '%s'. To attempt to relink them, try link_orphan_subjects",
                len(self.orphan_chunks),
                subject_set.display_name,
                )

        # known issue in pandas < 2.0, dtypes are changed with update, save the dtypes and restore them
        self.chunks = self.chunks.astype(CHUNKS_DTYPES)
        self.chunks.to_csv(self.chunks_file)
        
        signal.signal(signal.SIGINT, original_sigint_handler)
        signal.signal(signal.SIGTERM, original_sigterm_handler)
        
    def exit_upload(self, *args, rec_orphan, sub_set):
        if len(self.subjects_metadata) + len(self.orphan_chunks) != 0:

            if len(self.subjects_metadata):
                self.chunks.update(pd.DataFrame(self.subjects_metadata).set_index("index"))
            
            if rec_orphan and len(self.orphan_chunks): 
                self.chunks.update(pd.DataFrame(self.orphan_chunks).set_index("index"))

                logger_annotations.warning(
                "%d chunks were uploaded but not linked to the subject set '%s'. To attempt to relink them, try link_orphan_subjects",
                len(self.orphan_chunks),
                sub_set,
                )

            # known issue in pandas < 2.0, dtypes are changed with update, save the dtypes and restore them
            self.chunks.astype(CHUNKS_DTYPES)
            self.chunks.to_csv(self.chunks_file)
        logger_annotations.warning('Signal interruption %s, exited gracefully', args[0])
        sys.exit(0)
        
        
    def link_orphan_subjects(
        self,
        chunks: str,
        project_id: int,
        set_name: str,
        zooniverse_login="",
        zooniverse_pwd="",
        ignore_errors: bool = False,       
        test_endpoint: bool = False,
        **kwargs
    ):
        """Attempts to link subjects that have been uploaded but not linked to a subject set in zooniverse
        from the CSV dataframe `chunks` to a zooniverse project (Attempts are made on chunks that have a zooniverse_id,
        a project_id and uploaded at True but no subject_set )

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
        :param ignore_errors: carry on with the upload even if a clip fails, the csv will be updated accordingly
        :type ignore_errors: bool, optional
        :param test_endpoint: run this command for tests, operations with zooniverse arefaked and considered succesfull
        :type test_endpoint: bool, optional
        """
        
        self.chunks_file = chunks
        self.get_credentials(zooniverse_login, zooniverse_pwd)

        metadata_location = os.path.join(self.chunks_file)
        try:
            self.chunks = pd.read_csv(metadata_location, index_col="index", dtype=CHUNKS_DTYPES)
        except:
            raise Exception(
                "cannot read chunk metadata from {}.".format(metadata_location)
            )

        assert_dataframe("chunks", self.chunks)
        assert_columns_presence(
            "chunks",
            self.chunks,
            {"recording_filename", "onset", "offset", "uploaded", "mp3", "zooniverse_id", "project_id", "subject_set"},
        )
        
        if test_endpoint:
            from .fake_panoptes import Panoptes, Project, Subject, SubjectSet, PanoptesAPIException, reset_tests
            reset_tests() 
        else:
            from panoptes_client import Panoptes, Project, Subject, SubjectSet
            from panoptes_client.panoptes import PanoptesAPIException

        Panoptes.connect(username=self.zooniverse_login, password=self.zooniverse_pwd)
        zooniverse_project = Project(project_id)

        subjects_metadata = []

        subject_set = None

        for ss in zooniverse_project.links.subject_sets:
            if ss.display_name == set_name:
                subject_set = ss

        if subject_set is None:
            subject_set = SubjectSet()
            subject_set.links.project = zooniverse_project
            subject_set.display_name = set_name
            subject_set.save()

        # select chunks that are uploaded, have an id and project but no set
        chunks_to_link = self.chunks[(self.chunks["uploaded"] == True) &
                                      (~self.chunks['zooniverse_id'].isnull()) &
                                      (~self.chunks['project_id'].isnull()) &
                                      (self.chunks['subject_set'].isnull())]
        chunks_to_link = chunks_to_link.to_dict(orient="index")

        if len(chunks_to_link) == 0:
            logger_annotations.warning("no orphan chunks to link.")
            return

        for chunk_index in chunks_to_link:
            chunk = chunks_to_link[chunk_index]

            logger_annotations.info(
                "linking chunk %s (%d,%d)",
                chunk["recording_filename"],
                chunk["onset"],
                chunk["offset"],
                )

            try:
                subject = Subject.find(chunk['zooniverse_id'])
                
            except Exception as e:
                logger_annotations.error(
                "Could not find subject %s. an exception has occured:\n%s \n%s",
                chunk['zooniverse_id'],
                str(e),
                chunk["offset"],
                traceback.format_exc(),
                )

                if ignore_errors:
                    continue
                else:
                    logger_annotations.error("subject linking halting here.")
                    break
                
            try: 
                retry_func(subject_set.add, PanoptesAPIException, 3, subjects=subject)                
            except PanoptesAPIException as e:
                logger_annotations.error(
                "failed to add subject %d to subject_set %s. an exception has occured:\n%s \n%s",
                chunk_index,
                subject_set.display_name,
                str(e),
                traceback.format_exc(),
                )
                if ignore_errors and str(e):
                    continue
                else:
                    logger_annotations.error("subject linking halting here.")
                    break

            chunk["index"] = chunk_index
            chunk["subject_set"] = str(subject_set.display_name)
            chunk["subject_set_id"] = subject_set.id
            subjects_metadata.append(chunk)

        if len(subjects_metadata):
            tmp = pd.DataFrame(subjects_metadata)
            logger_annotations.info('%s \n%s', tmp.columns, tmp)
            self.chunks.update(pd.DataFrame(subjects_metadata).set_index("index"))

            # known issue in pandas < 2.0, dtypes are changed with update, save the dtypes and restore them
            self.chunks = self.chunks.astype(CHUNKS_DTYPES)
            self.chunks.to_csv(self.chunks_file)
        logger_annotations.info('linked %d/%d subjects', len(subjects_metadata), len(chunks_to_link))
        
    def reset_orphan_subjects(
        self,
        chunks: str,
        **kwargs
    ):
        """Look for orphan subjects and considers them to be not uploaded, This is to be done either if the oprhan
        subjects were deleted from zooniverse or if they are not usable anymore. The next upload will try to push 
        them to zooniverse as new subjects.

        :param chunks: path to the chunk CSV dataframe
        :type chunks: [type]
        """
        
        self.chunks_file = chunks

        metadata_location = os.path.join(self.chunks_file)
        try:
            self.chunks = pd.read_csv(metadata_location, index_col="index", dtype=CHUNKS_DTYPES)
        except:
            raise Exception(
                "cannot read chunk metadata from {}.".format(metadata_location)
            )

        assert_dataframe("chunks", self.chunks)
        assert_columns_presence(
            "chunks",
            self.chunks,
            {"recording_filename", "onset", "offset", "uploaded", "mp3", "zooniverse_id", "project_id", "subject_set"},
        )

        # select chunks that are uploaded, have an id and project but no set
        selection = ((self.chunks["uploaded"] == True) &
                      (~self.chunks['zooniverse_id'].isnull()) &
                      (~self.chunks['project_id'].isnull()) &
                      (self.chunks['subject_set'].isnull()))

        self.chunks.loc[selection , ['uploaded','zooniverse_id','project_id']] = (False, pd.NA, pd.NA)

        nb_reset = selection[selection == True]
        if nb_reset.shape[0]:
            self.chunks.to_csv(self.chunks_file)
            
            logger_annotations.info('reset %d orphan subjects', nb_reset)
        else:
            logger_annotations.info('no orphan subject to reset')

    def retrieve_classifications(
        self,
        destination: str,
        project_id: int,
        zooniverse_login: str = "",
        zooniverse_pwd: str = "",
        chunks: List[str] = [],
        test_endpoint: bool = False,
        **kwargs
    ):

        """Retrieve classifications from Zooniverse as a CSV dataframe.
        They will be matched with the original chunks metadata if the path one 
        or more chunk metadata files is provided.

        :param destination: output CSV dataframe destination
        :type destination: str
        :param project_id: zooniverse project id
        :type project_id: int
        :param zooniverse_login: zooniverse login. If not specified, the program attempts to get it from the environment variable ``ZOONIVERSE_LOGIN`` instead, defaults to ''
        :type zooniverse_login: str, optional
        :param zooniverse_pwd: zooniverse password. If not specified, the program attempts to get it from the environment variable ``ZOONIVERSE_PWD`` instead, defaults to ''
        :type zooniverse_pwd: str, optional
        :param chunks: the list of chunk metadata files to match the classifications to. If provided, only the classifications that have a match will be returned.
        :type chunks: List[str], optional
        """
        self.get_credentials(zooniverse_login, zooniverse_pwd)

        # if used in tests, use the fake functions
        if test_endpoint:
            from .fake_panoptes import Panoptes, Project, Classification
        else:
            from panoptes_client import Panoptes, Project, Classification
            
        Panoptes.connect(username=self.zooniverse_login, password=self.zooniverse_pwd)
        project = Project(project_id)

        answers_translation_table = []
        for workflow in project.links.workflows:
            workflow_id = workflow.id
            for task_id in workflow.tasks:
                n = 0
                for answer in workflow.tasks[task_id]["answers"]:
                    answers_translation_table.append(
                        {
                            "workflow_id": str(workflow_id),
                            "task_id": str(task_id),
                            "answer_id": str(n),
                            "answer": answer["label"],
                        }
                    )
                    n += 1

        answers_translation_table = pd.DataFrame(answers_translation_table)

        classifications = []
        for c in Classification.where(
            scope="project", page_size=1000, project_id=project_id
        ):
            classifications.append(c.raw)

        classifications = pd.DataFrame(classifications)
        classifications["user_id"] = classifications["links"].apply(lambda s: s["user"])
        classifications["subject_id"] = (
            classifications["links"].apply(lambda s: s["subjects"][0]).astype(int)
        )
        classifications["workflow_id"] = classifications["links"].apply(
            lambda s: s["workflow"]
        )
        classifications["tasks"] = classifications["annotations"].apply(
            lambda s: [(str(r["task"]), str(r["value"])) for r in s]
        )
        classifications = classifications.explode("tasks")
        classifications["task_id"] = classifications["tasks"].str[0]
        classifications["answer_id"] = classifications["tasks"].str[1]
        classifications.drop(columns=["tasks"], inplace=True)

        classifications = classifications[
            ["id", "user_id", "subject_id", "task_id", "answer_id", "workflow_id"]
        ]
        classifications = classifications.merge(
            answers_translation_table,
            left_on=["workflow_id", "task_id", "answer_id"],
            right_on=["workflow_id", "task_id", "answer_id"],
        )

        if chunks:
            chunks = pd.concat([pd.read_csv(f) for f in chunks])

            classifications = classifications.merge(
                chunks, left_on="subject_id", right_on="zooniverse_id"
            )

        classifications.set_index("id").to_csv(destination)

    def run(self, action, **kwargs):
        if action == "extract-chunks":
            return self.extract_chunks(**kwargs)
        elif action == "upload-chunks":
            return self.upload_chunks(**kwargs)
        elif action == "link-orphan-subjects":
            return self.link_orphan_subjects(**kwargs)
        elif action == "reset-orphan-subjects":
            return self.reset_orphan_subjects(**kwargs)
        elif action == "retrieve-classifications":
            return self.retrieve_classifications(**kwargs)

    @staticmethod
    def setup_parser(parser):
        subparsers = parser.add_subparsers(help="action", dest="action")

        """extract chunks parser"""
        parser_extraction = subparsers.add_parser(
            "extract-chunks",
            help="extract chunks to <destination>, and exports the metadata inside of this directory",
        )
        parser_extraction.add_argument("path", help="path to the dataset")
        parser_extraction.add_argument(
            "--keyword", help="export keyword", required=True
        )
        parser_extraction.add_argument(
            "--chunks-length",
            help="chunk length (in milliseconds). if <= 0, the segments will not be split into chunks (default value: 0)",
            type=int,
            default=0,
        )
        parser_extraction.add_argument(
            "--chunks-min-amount",
            help="minimum amount of chunks to extract from a segment (default value: 1)",
            default=1,
        )
        parser_extraction.add_argument(
            "--spectrogram",
            help="the extraction generates a png spectrogram (default False)",
            action="store_true",
        )
        parser_extraction.add_argument(
            "--segments", help="path to the input segments dataframe", required=True
        )
        parser_extraction.add_argument(
            "--destination", help="destination", required=True
        )
        parser_extraction.add_argument(
            "--profile",
            help="Recording profile to extract the audio clips from. If not specified, raw recordings will be used",
            default="",
        )
        parser_extraction.add_argument(
            "--threads", help="how many threads to run on", default=0, type=int
        )
        
        
        """upload subjects parser"""
        parser_upload = subparsers.add_parser(
            "upload-chunks", help="upload chunks and updates chunk state"
        )
        parser_upload.add_argument(
            "--chunks", help="path to the chunk CSV dataframe", required=True
        )
        parser_upload.add_argument(
            "--project-id", help="zooniverse project id", required=True, type=int
        )
        parser_upload.add_argument(
            "--set-name", help="subject set display name", required=True
        )
        parser_upload.add_argument(
            "--amount",
            help="amount of chunks to upload",
            required=False,
            type=int,
            default=1000,
        )
        parser_upload.add_argument(
            "--zooniverse-login",
            help="zooniverse login. If not specified, the program attempts to get it from the environment variable ZOONIVERSE_LOGIN instead",
            default="",
        )
        parser_upload.add_argument(
            "--zooniverse-pwd",
            help="zooniverse password. If not specified, the program attempts to get it from the environment variable ZOONIVERSE_PWD instead",
            default="",
        )
        parser_upload.add_argument(
            "--ignore-errors",
            help="keep uploading even when a subject fails to upload for some reason",
            action="store_true",
        )
        parser_upload.add_argument(
            "--record-orphan",
            help="list correctly create subjects as uploaded even if linking to a subject set failed",
            action="store_true",
        )
        
        
        """linking orphan subjects parser"""
        parser_link_subjects = subparsers.add_parser(
            "link-orphan-subjects", help="upload chunks and updates chunk state"
        )
        parser_link_subjects.add_argument(
            "--chunks", help="path to the chunk CSV dataframe", required=True
        )
        parser_link_subjects.add_argument(
            "--project-id", help="zooniverse project id", required=True, type=int
        )
        parser_link_subjects.add_argument(
            "--set-name", help="subject set display name", required=True
        )
        parser_link_subjects.add_argument(
            "--zooniverse-login",
            help="zooniverse login. If not specified, the program attempts to get it from the environment variable ZOONIVERSE_LOGIN instead",
            default="",
        )
        parser_link_subjects.add_argument(
            "--zooniverse-pwd",
            help="zooniverse password. If not specified, the program attempts to get it from the environment variable ZOONIVERSE_PWD instead",
            default="",
        )
        parser_link_subjects.add_argument(
            "--ignore-errors",
            help="keep uploading even when a subject fails to upload for some reason",
            action="store_true",
        )
        
        
        """reset orphan subjects parser"""
        parser_reset_orphan = subparsers.add_parser(
            "reset-orphan-subjects", help="upload chunks and updates chunk state"
        )
        parser_reset_orphan.add_argument(
            "--chunks", help="path to the chunk CSV dataframe", required=True
        )
        
        
        """retrieve classifications parser"""
        parser_retrieve = subparsers.add_parser(
            "retrieve-classifications",
            help="retrieve classifications and save them as <destination>",
        )
        parser_retrieve.add_argument(
            "--destination", help="output CSV dataframe destination", required=True
        )
        parser_retrieve.add_argument(
            "--project-id", help="zooniverse project id", required=True, type=int
        )
        parser_retrieve.add_argument(
            "--zooniverse-login",
            help="zooniverse login. If not specified, the program attempts to get it from the environment variable ZOONIVERSE_LOGIN instead",
            default="",
        )
        parser_retrieve.add_argument(
            "--zooniverse-pwd",
            help="zooniverse password. If not specified, the program attempts to get it from the environment variable ZOONIVERSE_PWD instead",
            default="",
        )
        parser_retrieve.add_argument(
            "--chunks", help="list of chunks", nargs="+", required=True
        )

