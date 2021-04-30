from collections import defaultdict
import pandas as pd
import re

class Converter:
    SPEAKER_ID_TO_TYPE = {
        'C1': 'OCH',
        'C2': 'OCH',
        'CHI': 'CHI',
        'CHI*': 'CHI',
        'FA0': 'FEM',
        'FA1': 'FEM',
        'FA2': 'FEM',
        'FA3': 'FEM',
        'FA4': 'FEM',
        'FA5': 'FEM',
        'FA6': 'FEM',
        'FA7': 'FEM',
        'FA8': 'FEM',
        'FC1': 'OCH',
        'FC2': 'OCH',
        'FC3': 'OCH',
        'MA0': 'MAL',
        'MA1': 'MAL',
        'MA2': 'MAL',
        'MA3': 'MAL',
        'MA4': 'MAL',
        'MA5': 'MAL',
        'MC1': 'OCH',
        'MC2': 'OCH',
        'MC3': 'OCH',
        'MC4': 'OCH',
        'MC5': 'OCH',
        'MI1': 'OCH',
        'MOT*': 'FEM',
        'OC0': 'OCH',
        'UC1': 'OCH',
        'UC2': 'OCH',
        'UC3': 'OCH',
        'UC4': 'OCH',
        'UC5': 'OCH',
        'UC6': 'OCH'
    }

class VtcConverter(Converter):

    SPEAKER_TYPE_TRANSLATION = defaultdict(lambda: 'NA', {
        'CHI': 'OCH',
        'KCHI': 'CHI',
        'FEM': 'FEM',
        'MAL':'MAL',
        'SPEECH': 'SPEECH'
    })

    @staticmethod
    def convert(filename: str, source_file: str = '') -> pd.DataFrame:
        rttm = pd.read_csv(
            filename,
            sep = " ",
            names = ['type', 'file', 'chnl', 'tbeg', 'tdur', 'ortho', 'stype', 'name', 'conf', 'unk']
        )

        df = rttm
        df['segment_onset'] = df['tbeg'].mul(1000).round().astype(int)
        df['segment_offset'] = (df['tbeg']+df['tdur']).mul(1000).round().astype(int)
        df['speaker_type'] = df['name'].map(VtcConverter.SPEAKER_TYPE_TRANSLATION)

        if source_file:
            df = df[df['file'] == source_file]

        df.drop(['type', 'file', 'chnl', 'tbeg', 'tdur', 'ortho', 'stype', 'name', 'conf', 'unk'], axis = 1, inplace = True)

        return df

class VcmConverter(Converter):

    SPEAKER_TYPE_TRANSLATION = defaultdict(lambda: 'NA', {
        'CHI': 'OCH',
        'CRY': 'CHI',
        'NCS': 'CHI',
        'CNS': 'CHI',
        'FEM': 'FEM',
        'MAL':'MAL',
        'SPEECH': 'SPEECH'
    })

    VCM_TRANSLATION = defaultdict(lambda: 'NA', {
        'CRY': 'Y',
        'NCS': 'N',
        'CNS': 'C',
        'OTH': 'J'
    })

    @staticmethod
    def convert(filename: str, source_file: str = '') -> pd.DataFrame:
        rttm = pd.read_csv(
            filename,
            sep = " ",
            names = ['type', 'file', 'chnl', 'tbeg', 'tdur', 'ortho', 'stype', 'name', 'conf', 'unk']
        )

        df = rttm
        df['segment_onset'] = df['tbeg'].mul(1000).round().astype(int)
        df['segment_offset'] = (df['tbeg']+df['tdur']).mul(1000).round().astype(int)
        df['speaker_type'] = df['name'].map(VcmConverter.SPEAKER_TYPE_TRANSLATION)
        df['vcm_type'] = df['name'].map(VcmConverter.VCM_TRANSLATION)

        if source_file:
            df = df[df['file'] == source_file]

        df.drop(['type', 'file', 'chnl', 'tbeg', 'tdur', 'ortho', 'stype', 'name', 'conf', 'unk'], axis = 1, inplace = True)

        return df

class AliceConverter(Converter):

    @staticmethod
    def convert(filename: str, source_file: str = '') -> pd.DataFrame:
        df = pd.read_csv(
            filename,
            sep = r"\s",
            names = ['file', 'phonemes', 'syllables', 'words'],
            engine = 'python'
        )

        if source_file:
            df = df[df['file'].str.contains(source_file)]

        matches = df['file'].str.extract(r"^(.*)_(?:0+)?([0-9]{1,})_(?:0+)?([0-9]{1,})\.wav$")
        df['recording_filename'] = matches[0]
        df['segment_onset'] = matches[1].astype(int)/10
        df['segment_offset'] = matches[2].astype(int)/10
        
        df.drop(columns = ['recording_filename', 'file'], inplace = True)

        return df

class ItsConverter(Converter):

    SPEAKER_TYPE_TRANSLATION = {
        'CHN': 'CHI',
        'CXN': 'OCH',
        'FAN': 'FEM',
        'MAN': 'MAL',
        'OLN': 'OLN',
        'TVN': 'TVN',
        'NON': 'NON',
        'SIL': 'SIL',
        'FUZ': 'FUZ',
        'TVF': 'TVF',
        'CXF': 'CXF',
        'NOF': 'NON',
        'OLF': 'OLN',
        'CHF': 'CHF',
        'MAF': 'MAF',
        'FAF': 'FEF'
    }

    @staticmethod
    def convert(filename: str, recording_num: int = None) -> pd.DataFrame:
        from lxml import etree

        xml = etree.parse(filename)

        recordings = xml.xpath('/ITS/ProcessingUnit/Recording' + ('[@num="{}"]'.format(recording_num) if recording_num else ''))
        timestamp_pattern = re.compile(r"^P(?:T?)(\d+(\.\d+)?)S$")

        def extract_from_regex(pattern, subject):
            match = pattern.search(subject)
            return match.group(1) if match else ''

        segments = []

        for recording in recordings:
            segs = recording.xpath('./Pause/Segment|./Conversation/Segment')
            for seg in segs:
                parent = seg.getparent()

                lena_block_number = parent.get('num')
                lena_block_type = 'pause' if parent.tag.lower() == 'pause' else parent.get('type')

                if not seg.get('conversationInfo'):
                    conversation_info = ['NA'] * 7
                else:
                    conversation_info = seg.get('conversationInfo').split('|')[1:-1]
                
                lena_conv_status = conversation_info[0]
                lena_response_count = conversation_info[3]
                lena_conv_turn_type = conversation_info[5]
                lena_conv_floor_type = conversation_info[6]

                onset = float(extract_from_regex(timestamp_pattern, seg.get('startTime')))
                offset = float(extract_from_regex(timestamp_pattern, seg.get('endTime')))

                words = 0
                for attr in ['femaleAdultWordCnt', 'maleAdultWordCnt']:
                    words += float(seg.get(attr, 0))

                utterances_count = 0
                for attr in ['femaleAdultUttCnt', 'maleAdultUttCnt', 'childUttCnt']:
                    utterances_count += float(seg.get(attr, 0))

                utterances_length = 0
                for attr in ['femaleAdultUttLen', 'maleAdultUttLen', 'childUttLen']:
                    utterances_length += float(extract_from_regex(timestamp_pattern, seg.get(attr, 'P0S')))

                non_speech_length = 0
                for attr in ['femaleAdultNonSpeechLen', 'maleAdultNonSpeechLen']:
                    non_speech_length += float(extract_from_regex(timestamp_pattern, seg.get(attr, 'P0S')))

                average_db = seg.get('average_dB', 0)
                peak_db = seg.get('peak_dB', 0)

                utterances = seg.xpath('./UTT')
                utterances = [dict(utt.attrib) for utt in utterances]

                if not utterances:
                    n = 1
                    while 'startUtt{}'.format(n) in seg.attrib:
                        start = 'startUtt{}'.format(n)
                        end = 'endUtt{}'.format(n)
                        utterances.append({
                            start: seg.attrib[start],
                            end: seg.attrib[end]
                        })
                        n = n + 1

                for utterance in utterances:
                    for c in list(utterance.keys()):
                        if 'startUtt' in c:
                            utterance['start'] = float(extract_from_regex(timestamp_pattern, utterance.pop(c)))
                        elif 'endUtt' in c:
                            utterance['end'] = float(extract_from_regex(timestamp_pattern, utterance.pop(c)))
                
                child_cry_vfx_len = float(extract_from_regex(timestamp_pattern, seg.get('childCryVfxLen', 'PT0S')))

                cries = []
                n = 1
                while 'startCry{}'.format(n) in seg.attrib:
                    start = 'startCry{}'.format(n)
                    end = 'endCry{}'.format(n)
                    cries.append({
                        'start': float(extract_from_regex(timestamp_pattern, seg.attrib[start])),
                        'end': float(extract_from_regex(timestamp_pattern, seg.attrib[end]))
                    })
                    n = n + 1

                vfxs = []
                n = 1
                while 'startVfx{}'.format(n) in seg.attrib:
                    start = 'startVfx{}'.format(n)
                    end = 'endVfx{}'.format(n)
                    vfxs.append({
                        'start': float(extract_from_regex(timestamp_pattern, seg.attrib[start])),
                        'end': float(extract_from_regex(timestamp_pattern, seg.attrib[end]))
                    })
                    n = n + 1

                segments.append({
                    'segment_onset': int(round(onset*1000)),
                    'segment_offset': int(round(offset*1000)),
                    'speaker_type': ItsConverter.SPEAKER_TYPE_TRANSLATION[seg.get('spkr')],
                    'words': words,
                    'lena_block_number': lena_block_number,
                    'lena_block_type': lena_block_type,
                    'lena_conv_status': lena_conv_status,
                    'lena_response_count': lena_response_count,
                    'lena_conv_turn_type': lena_conv_turn_type,
                    'lena_conv_floor_type': lena_conv_floor_type,
                    'utterances_count': utterances_count,
                    'utterances_length': int(utterances_length*1000),
                    'average_db': average_db,
                    'peak_db': peak_db,
                    'utterances': utterances,
                    'non_speech_length': int(non_speech_length*1000),
                    'child_cry_vfx_len': int(child_cry_vfx_len*1000),
                    'cries': cries,
                    'vfxs': vfxs
                })
                
        df = pd.DataFrame(segments)
        
        return df

class TextGridConverter(Converter):

    @staticmethod
    def convert(filename: str) -> pd.DataFrame:
        import pympi
        textgrid = pympi.Praat.TextGrid(filename)

        def ling_type(s):
            s = str(s)

            a, b = ('0' in s, '1' in s)
            if a^b:
                return '0' if a else '1' 
            else:
                return 'NA'

        segments = []
        for tier in textgrid.tiers:
            for interval in tier.intervals:
                tier_name = tier.name.strip()

                if tier_name == 'Autre':
                    continue

                if interval[2] == "":
                    continue

                segment = {
                    'segment_onset': int(round(1000*float(interval[0]))),
                    'segment_offset': int(round(1000*float(interval[1]))),
                    'speaker_id': tier_name,
                    'ling_type': ling_type(interval[2]),
                    'speaker_type': Converter.SPEAKER_ID_TO_TYPE[tier_name] if tier_name in Converter.SPEAKER_ID_TO_TYPE else 'NA'
                }

                segments.append(segment)

        return pd.DataFrame(segments)

class EafConverter(Converter):
    
    @staticmethod
    def convert(filename: str) -> pd.DataFrame:
        import pympi
        eaf = pympi.Elan.Eaf(filename)

        segments = {}
        for tier_name in eaf.tiers:
            annotations = eaf.tiers[tier_name][0]

            if tier_name not in Converter.SPEAKER_ID_TO_TYPE and len(annotations) > 0:
                print("warning: unknown tier '{}' will be ignored in '{}'".format(tier_name, filename))
                continue

            for aid in annotations:
                (start_ts, end_ts, value, svg_ref) = annotations[aid]
                (start_t, end_t) = (eaf.timeslots[start_ts], eaf.timeslots[end_ts])

                segment = {
                    'segment_onset': int(round(start_t)),
                    'segment_offset': int(round(end_t)),
                    'speaker_id': tier_name,
                    'speaker_type': Converter.SPEAKER_ID_TO_TYPE[tier_name] if tier_name in Converter.SPEAKER_ID_TO_TYPE else 'NA',
                    'vcm_type': 'NA',
                    'lex_type': 'NA',
                    'mwu_type': 'NA',
                    'addressee': 'NA',
                    'transcription': value if value != '0' else '0.',
                    'words':'NA'
                }

                segments[aid] = segment

        for tier_name in eaf.tiers:
            if '@' in tier_name:
                label, ref = tier_name.split('@')
            else:
                label, ref = tier_name, None

            reference_annotations = eaf.tiers[tier_name][1]

            if ref not in Converter.SPEAKER_ID_TO_TYPE:
                continue

            for aid in reference_annotations:
                (ann, value, prev, svg) = reference_annotations[aid]

                ann = aid
                parentTier = eaf.tiers[eaf.annotations[ann]]
                while 'PARENT_REF' in parentTier[2] and parentTier[2]['PARENT_REF'] and len(parentTier[2]) > 0:
                    ann = parentTier[1][ann][0]
                    parentTier = eaf.tiers[eaf.annotations[ann]]

                if ann not in segments:
                    print("warning: annotation '{}' not found in segments for '{}'".format(ann, filename))
                    continue
                
                segment = segments[ann]

                if label == 'lex':
                    segment['lex_type'] = value
                elif label == 'mwu':
                    segment['mwu_type'] = value
                elif label == 'xds':
                    segment['addressee'] = value
                elif label == 'vcm':
                    segment['vcm_type'] = value

        return pd.DataFrame(segments.values())

def ChatConverter(Converter):
    @staticmethod
    def convert(filename: str,
    speaker_id_to_type: dict = None,
    addressee_table: dict = None) -> pd.DataFrame:
        import pylangacq

        if speaker_id_to_type is None:
            speaker_id_to_type = {
                'MOT': 'FEM',
                'FAT': 'MAL',
                'SIS': 'OCH'
            }

        if addressee_table is None:
            addressee_table = defaultdict(lambda: 'NA', {
                'MOT': 'A',
                'FAT': 'A',
                'SIS': 'C',
                'CHI': 'T'
            })

        reader = pylangacq.Reader.from_files([filename])
        df = pd.DataFrame(reader.utterances())

        ### extract tiers
        df['tiers'] = df['tiers'].apply(lambda d: {k.replace('%', ''): d[k] for k in d.keys()})
        df = pd.concat([df.drop(['tiers'], axis = 1), df['tiers'].apply(pd.Series)], axis = 1)

        df['segment_onset'] = df['time_marks'].apply(lambda tm: tm[0] if tm else 'NA')
        df['segment_offset'] = df['time_marks'].apply(lambda tm: tm[1] if tm else 'NA')

        df['speaker_id'] = df['participant']
        df['speaker_type'] = df['speaker_id'].replace(speaker_id_to_type)

        df['transcription'] = df['tokens'].apply(lambda l: ' '.join([t['word'] for t in l]))

        if 'add' in df.columns:
            df['addressee'] = df['add'].str.split(',')\
                .apply(lambda l: ','.join(sorted([addressee_table[x.strip()] for x in l])))

        df = df[(df['segment_onset'] != 'NA') & (df['segment_offset'] != 'NA')]
        df.drop(columns = ['tokens', 'time_marks'], inplace = True)
        df.fillna('NA', inplace = True)

        return df