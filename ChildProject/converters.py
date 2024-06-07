from collections import defaultdict
import pandas as pd
import re
from enum import Enum
from .pipelines.metricsFunctions import voc_speaker

converters = {}

class Formats(Enum):
    CSV = 'csv'
    VTC = 'vtc_rttm'
    VCM = 'vcm_rttm'
    ALICE = 'alice'
    ITS = 'its'
    TEXTGRID = 'TextGrid'
    EAF = 'eaf'
    CHA = 'cha'

class AnnotationConverter:
    SPEAKER_ID_TO_TYPE = defaultdict(
        lambda: "NA",
        {
            "C1": "OCH",
            "C2": "OCH",
            "CHI": "CHI",
            "CHI*": "CHI",
            "FA0": "FEM",
            "FA1": "FEM",
            "FA2": "FEM",
            "FA3": "FEM",
            "FA4": "FEM",
            "FA5": "FEM",
            "FA6": "FEM",
            "FA7": "FEM",
            "FA8": "FEM",
            "FC1": "OCH",
            "FC2": "OCH",
            "FC3": "OCH",
            "MA0": "MAL",
            "MA1": "MAL",
            "MA2": "MAL",
            "MA3": "MAL",
            "MA4": "MAL",
            "MA5": "MAL",
            "MC1": "OCH",
            "MC2": "OCH",
            "MC3": "OCH",
            "MC4": "OCH",
            "MC5": "OCH",
            "MI1": "OCH",
            "MOT*": "FEM",
            "OC0": "OCH",
            "UC1": "OCH",
            "UC2": "OCH",
            "UC3": "OCH",
            "UC4": "OCH",
            "UC5": "OCH",
            "UC6": "OCH",
            "UA1": "NA",
            "UA2": "NA",
            "UA3": "NA",
            "UA4": "NA",
            "UA5": "NA",
            "UA6": "NA",
            "EE1": "NA",
            "EE2": "NA",
            "FAE": "NA",
            "MAE": "NA",
            "FCE": "NA",
            "MCE": "NA",
        },
    )

    THREAD_SAFE = True

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        converters[cls.FORMAT] = cls


class CsvConverter(AnnotationConverter):
    FORMAT = Formats.CSV.value

    @staticmethod
    def convert(filename: str, filter: str="", **kwargs) -> pd.DataFrame:
        return pd.read_csv(filename)


class VtcConverter(AnnotationConverter):
    FORMAT = Formats.VTC.value

    SPEAKER_TYPE_TRANSLATION = defaultdict(
        lambda: "NA", {"CHI": "OCH", "KCHI": "CHI", "FEM": "FEM", "MAL": "MAL"}
    )

    @staticmethod
    def convert(filename: str, source_file: str = "", **kwargs) -> pd.DataFrame:
        rttm = pd.read_csv(
            filename,
            sep=" ",
            names=[
                "type",
                "file",
                "chnl",
                "tbeg",
                "tdur",
                "ortho",
                "stype",
                "name",
                "conf",
                "unk",
            ],
            dtype={'type': str, "file" : str, 'stype':str}
        )

        n_recordings = len(rttm["file"].unique())
        if  n_recordings > 1 and not source_file:
            print(
                f"""WARNING: {filename} contains annotations from {n_recordings} different audio files, """
                """but no filter was specified which means all of these annotations will be imported\n"""
                """as if they belonged to the same recording. Please make sure this is the intended behavior """
                """(it probably isn't)."""
            )

        df = rttm
        df["segment_onset"] = df["tbeg"].mul(1000).round().astype(int)
        df["segment_offset"] = (df["tbeg"] + df["tdur"]).mul(1000).round().astype(int)
        df["speaker_type"] = df["name"].map(VtcConverter.SPEAKER_TYPE_TRANSLATION)

        if source_file:
            df = df[df["file"] == source_file]

        df.drop(
            [
                "type",
                "file",
                "chnl",
                "tbeg",
                "tdur",
                "ortho",
                "stype",
                "name",
                "conf",
                "unk",
            ],
            axis=1,
            inplace=True,
        )

        return df


class VcmConverter(AnnotationConverter):
    FORMAT = Formats.VCM.value

    SPEAKER_TYPE_TRANSLATION = defaultdict(
        lambda: "NA",
        {
            "CHI": "OCH",
            "CRY": "CHI",
            "NCS": "CHI",
            "CNS": "CHI",
            "FEM": "FEM",
            "MAL": "MAL",

        },
    )

    VCM_TRANSLATION = defaultdict(
        lambda: "NA", {"CRY": "Y", "NCS": "N", "CNS": "C", "OTH": "J"}
    )

    @staticmethod
    def convert(filename: str, source_file: str = "", **kwargs) -> pd.DataFrame:
        rttm = pd.read_csv(
            filename,
            sep=" ",
            names=[
                "type",
                "file",
                "chnl",
                "tbeg",
                "tdur",
                "ortho",
                "stype",
                "name",
                "conf",
                "unk",
            ],
        )

        df = rttm
        df["segment_onset"] = df["tbeg"].mul(1000).round().astype(int)
        df["segment_offset"] = (df["tbeg"] + df["tdur"]).mul(1000).round().astype(int)
        df["speaker_type"] = df["name"].map(VcmConverter.SPEAKER_TYPE_TRANSLATION)
        df["vcm_type"] = df["name"].map(VcmConverter.VCM_TRANSLATION)

        if source_file:
            df = df[df["file"] == source_file]

        df.drop(
            [
                "type",
                "file",
                "chnl",
                "tbeg",
                "tdur",
                "ortho",
                "stype",
                "name",
                "conf",
                "unk",
            ],
            axis=1,
            inplace=True,
        )

        return df


class AliceConverter(AnnotationConverter):
    FORMAT = Formats.ALICE.value

    @staticmethod
    def convert(filename: str, source_file: str = "", **kwargs) -> pd.DataFrame:
        df = pd.read_csv(
            filename,
            sep=r"\s",
            names=["file", "phonemes", "syllables", "words"],
            engine="python",
        )

        n_recordings = len(df["file"].str.split('_').apply(lambda x: x[:-2]).str.join('_').unique())
        if  n_recordings > 1 and not source_file:
            print(
                f"""WARNING: {filename} contains annotations from {n_recordings} different audio files, """
                """but no filter was specified which means all of these annotations will be imported.\n"""
                """as if they belonged to the same recording. Please make sure this is the intended behavior """
                """(it probably isn't)."""
            )

        if source_file:
            df = df[df["file"].str.contains(source_file)]

        matches = df["file"].str.extract(
            r"^(.*)_(?:0+)?([0-9]{1,})_(?:0+)?([0-9]{1,})\.wav$"
        )
        df["recording_filename"] = matches[0]
        df["segment_onset"] = matches[1].astype(int) / 10
        df["segment_offset"] = matches[2].astype(int) / 10

        df.drop(columns=["recording_filename", "file"], inplace=True)

        return df


class ItsConverter(AnnotationConverter):
    FORMAT = Formats.ITS.value

    SPEAKER_TYPE_TRANSLATION = defaultdict(
        lambda: "NA", {"CHN": "CHI", "CXN": "OCH", "FAN": "FEM", "MAN": "MAL"}
    )

    @staticmethod
    def convert(filename: str, recording_num: int = None, **kwargs) -> pd.DataFrame:
        from lxml import etree

        xml = etree.parse(filename)

        recordings = xml.xpath(
            "/ITS/ProcessingUnit/Recording"
            + ('[@num="{}"]'.format(recording_num) if recording_num else "")
        )
        timestamp_pattern = re.compile(r"^P(?:T?)(\d+(\.\d+)?)S$")

        def extract_from_regex(pattern, subject):
            match = pattern.search(subject)
            return match.group(1) if match else ""

        segments = []

        for recording in recordings:
            segs = recording.xpath("./Pause/Segment|./Conversation/Segment")
            for seg in segs:
                parent = seg.getparent()

                lena_block_number = int(parent.get("num"))
                lena_block_type = (
                    "pause" if parent.tag.lower() == "pause" else parent.get("type")
                )

                if not seg.get("conversationInfo"):
                    conversation_info = ["NA"] * 7
                else:
                    conversation_info = seg.get("conversationInfo").split("|")[1:-1]

                lena_conv_status = conversation_info[0]
                lena_response_count = conversation_info[3]
                lena_conv_turn_type = conversation_info[5]
                lena_conv_floor_type = conversation_info[6]

                onset = float(
                    extract_from_regex(timestamp_pattern, seg.get("startTime"))
                )
                offset = float(
                    extract_from_regex(timestamp_pattern, seg.get("endTime"))
                )

                words = 0
                for attr in ["femaleAdultWordCnt", "maleAdultWordCnt"]:
                    words += float(seg.get(attr, 0))

                utterances_count = 0
                for attr in ["femaleAdultUttCnt", "maleAdultUttCnt", "childUttCnt"]:
                    utterances_count += float(seg.get(attr, 0))

                utterances_length = 0
                for attr in ["femaleAdultUttLen", "maleAdultUttLen", "childUttLen"]:
                    utterances_length += float(
                        extract_from_regex(timestamp_pattern, seg.get(attr, "P0S"))
                    )

                non_speech_length = 0
                for attr in ["femaleAdultNonSpeechLen", "maleAdultNonSpeechLen"]:
                    non_speech_length += float(
                        extract_from_regex(timestamp_pattern, seg.get(attr, "P0S"))
                    )

                average_db = float(seg.get("average_dB", 0))
                peak_db = float(seg.get("peak_dB", 0))

                utterances = seg.xpath("./UTT")
                utterances = [dict(utt.attrib) for utt in utterances]

                if not utterances:
                    n = 1
                    while "startUtt{}".format(n) in seg.attrib:
                        start = "startUtt{}".format(n)
                        end = "endUtt{}".format(n)
                        utterances.append(
                            {start: seg.attrib[start], end: seg.attrib[end]}
                        )
                        n = n + 1

                for utterance in utterances:
                    for c in list(utterance.keys()):
                        if "startUtt" in c:
                            utterance["start"] = float(
                                extract_from_regex(timestamp_pattern, utterance.pop(c))
                            )
                        elif "endUtt" in c:
                            utterance["end"] = float(
                                extract_from_regex(timestamp_pattern, utterance.pop(c))
                            )

                child_cry_vfx_len = float(
                    extract_from_regex(
                        timestamp_pattern, seg.get("childCryVfxLen", "PT0S")
                    )
                )

                cries = []
                n = 1
                while "startCry{}".format(n) in seg.attrib:
                    start = "startCry{}".format(n)
                    end = "endCry{}".format(n)
                    cries.append(
                        {
                            "start": float(
                                extract_from_regex(timestamp_pattern, seg.attrib[start])
                            ),
                            "end": float(
                                extract_from_regex(timestamp_pattern, seg.attrib[end])
                            ),
                        }
                    )
                    n = n + 1

                vfxs = []
                n = 1
                while "startVfx{}".format(n) in seg.attrib:
                    start = "startVfx{}".format(n)
                    end = "endVfx{}".format(n)
                    vfxs.append(
                        {
                            "start": float(
                                extract_from_regex(timestamp_pattern, seg.attrib[start])
                            ),
                            "end": float(
                                extract_from_regex(timestamp_pattern, seg.attrib[end])
                            ),
                        }
                    )
                    n = n + 1

                segments.append(
                    {
                        "segment_onset": int(round(onset * 1000)),
                        "segment_offset": int(round(offset * 1000)),
                        "speaker_type": ItsConverter.SPEAKER_TYPE_TRANSLATION[
                            seg.get("spkr")
                        ],
                        "lena_speaker": seg.get("spkr"),
                        "words": words,
                        "lena_block_number": lena_block_number,
                        "lena_block_type": lena_block_type,
                        "lena_conv_status": lena_conv_status,
                        "lena_response_count": lena_response_count,
                        "lena_conv_turn_type": lena_conv_turn_type,
                        "lena_conv_floor_type": lena_conv_floor_type,
                        "utterances_count": utterances_count,
                        "utterances_length": int(utterances_length * 1000),
                        "average_db": average_db,
                        "peak_db": peak_db,
                        "utterances": utterances,
                        "non_speech_length": int(non_speech_length * 1000),
                        "child_cry_vfx_len": int(child_cry_vfx_len * 1000),
                        "cries": cries,
                        "vfxs": vfxs,
                    }
                )

        df = pd.DataFrame(segments)

        return df


class TextGridConverter(AnnotationConverter):
    FORMAT = Formats.TEXTGRID.value

    @staticmethod
    def convert(filename: str, filter=None, **kwargs) -> pd.DataFrame:
        import pympi

        textgrid = pympi.Praat.TextGrid(filename)

        def ling_type(s):
            s = str(s)

            a, b = ("0" in s, "1" in s)
            if a ^ b:
                return "0" if a else "1"
            else:
                return "NA"

        segments = []
        for tier in textgrid.tiers:
            for interval in tier.intervals:
                tier_name = tier.name.strip()

                if tier_name == "Autre":
                    continue

                if interval[2] == "":
                    continue

                segment = {
                    "segment_onset": int(round(1000 * float(interval[0]))),
                    "segment_offset": int(round(1000 * float(interval[1]))),
                    "speaker_id": tier_name,
                    "ling_type": ling_type(interval[2]),
                    "speaker_type": AnnotationConverter.SPEAKER_ID_TO_TYPE[tier_name],
                }

                segments.append(segment)

        return pd.DataFrame(segments)


class EafConverter(AnnotationConverter):
    FORMAT = Formats.EAF.value

    @staticmethod
    def convert(filename: str, filter=None, **kwargs) -> pd.DataFrame:
        import pympi

        eaf = pympi.Elan.Eaf(filename)

        segments = {}
        for tier_name in eaf.tiers:
            annotations = eaf.tiers[tier_name][0]

            if (
                tier_name not in AnnotationConverter.SPEAKER_ID_TO_TYPE
                and len(annotations) > 0
            ):
                print(
                    "warning: unknown tier '{}' will be ignored in '{}'".format(
                        tier_name, filename
                    )
                )
                continue

            for aid in annotations:
                (start_ts, end_ts, value, svg_ref) = annotations[aid]
                (start_t, end_t) = (eaf.timeslots[start_ts], eaf.timeslots[end_ts])

                segment = {
                    "segment_onset": int(round(start_t)),
                    "segment_offset": int(round(end_t)),
                    "speaker_id": tier_name,
                    "speaker_type": AnnotationConverter.SPEAKER_ID_TO_TYPE[tier_name],
                    "vcm_type": "NA",
                    "lex_type": "NA",
                    "mwu_type": "NA",
                    "addressee": "NA",
                    "transcription": value if value != "0" else "0.",
                    "words": "NA",
                }

                segments[aid] = segment

        for tier_name in eaf.tiers:
            if "@" in tier_name:
                label, ref = tier_name.split("@")
            else:
                label, ref = tier_name, None

            reference_annotations = eaf.tiers[tier_name][1]

            if ref not in AnnotationConverter.SPEAKER_ID_TO_TYPE:
                continue

            for aid in reference_annotations:
                (ann, value, prev, svg) = reference_annotations[aid]

                ann = aid
                parentTier = eaf.tiers[eaf.annotations[ann]]
                while (
                    "PARENT_REF" in parentTier[2]
                    and parentTier[2]["PARENT_REF"]
                    and len(parentTier[2]) > 0
                ):
                    ann = parentTier[1][ann][0]
                    parentTier = eaf.tiers[eaf.annotations[ann]]

                if ann not in segments:
                    print(
                        "warning: annotation '{}' not found in segments for '{}'".format(
                            ann, filename
                        )
                    )
                    continue

                segment = segments[ann]

                if label == "lex":
                    segment["lex_type"] = value
                elif label == "mwu":
                    segment["mwu_type"] = value
                elif label == "xds":
                    segment["addressee"] = value
                elif label == "vcm":
                    segment["vcm_type"] = value
                elif label == "msc":
                    segment["msc_type"] = value
                elif label in kwargs["new_tiers"]:
                    segment[label] = value

        return pd.DataFrame(segments.values())


class ChatConverter(AnnotationConverter):
    FORMAT = Formats.CHA.value
    THREAD_SAFE = False

    SPEAKER_ROLE_TO_TYPE = defaultdict(
        lambda: "NA",
        {
            "Target_Child": "CHI",  # Use of this role is very important for CHILDES and PhonBank transcripts, because it allows users to search and analyze the output from the children who are the focus of many of the studies.
            "Target_Adult": "NA",  # This role serves a similar function to Target_Child by making it clear who which speaker was at the focus of the data collection.
            "Child": "OCH",  # This role is used mostly in transcripts studying large groups of children, when it is not easy to determine whether a child is a boy or girl or perhaps a relative.
            "Mother": "FEM",  # This should be the mother of the Target_Child.
            "Father": "MAL",  # This should be the father of the Target_Child.
            "Brother": "OCH",  # This should be a brother of the Target_Child.
            "Sister": "OCH",  # This should be a sister of the Target_Child.
            "Sibling": "OCH",  # This should be a sibling of the Target_Child.
            "Grandfather": "MAL",  # This should be the grandfather of the Target_Child.  Further details such as Paternal_Grandfather can be placed into the Specific Role field.
            "Grandmother": "FEM",  # This should be the grandmother of the Target_Child.  Further details such as Paternal_Grandmother can be placed into the Specific Role field.
            "Relative": "NA",  # This role is designed to include all other relations, including Aunt, Uncle, Cousin, Father_in_Law etc. which can then be entered into the Specific Role field.
            "Participant": "CHI",  # This is the generic role for adult participants in interviews and other conversations.  Usually, these are coded as having a Participant and an Investigator.  Other forms of this role include Patient, Informant, and Subject which can be listed in the Specific Role field or else just omitted.
            "Investigator": "NA",  #  Other terms for this role can be listed in the Specific Roles.  These include Researcher, Clinician, Therapist, Observer, Camera_Operator, and so on.
            "Partner": "NA",  #  This is the role for the person accompanying the Participant to the interview or conversation.
            "Boy": "OCH",  #  This is a generic role.
            "Girl": "OCH",  # This is a generic role.
            "Adult": "NA",  #  This is a very generic role for use when little else is known.
            "Teenager": "NA",  # This is a generic role.
            "Male": "MAL",  # Use this role when all we know is that the participant is an adult male.
            "Female": "FEM",  #  Use this role when all we know is that the participant is an adult female.
            "Visitor": "NA",  #  This role assumes that the visitor is coming to a conversation in the home.
            "Friend": "OCH",  # This is a role for a Friend of the target participants.
            "Playmate": "OCH",  #  This is a role for a child that the Target_Child plays with.
            "Caretaker": "NA",  #  This person takes care of the child. Other names for the Specific Role field include Housekeeper, Nursemaid, or Babysitter.
            "Environment": "NA",  # This role is used in the SBCSAE corpus.
            "Group": "NA",  # This role is used when transcribing simultaneous productions from a whole group.
            "Unidentified": "NA",  #  This is a role for unidentifiable participants.
            "Uncertain": "NA",  # This role can be used when it is not clear who produced an utterance.
            "Other": "NA",  # This is a generic role.  When it is used, there should be further specification in the Specific Role field. Roles defined by jobs such as Technician, Patron, Policeman, etc can be listed as Other and the details given in the Specific Role field.
            "Text": "NA",  # This role is used for written segments of TalkBank.
            "Media": "NA",  #  This role is used for speech from televisions, computers, or talking toys.
            "PlayRole": "NA",  # This role is used when speakers pretend to be something, such as an animal or another person.
            "LENA": "NA",  # This role is used in HomeBank LENA recordings.  The specific LENA role is then listed in the Specific Role field.
            "Justice": "NA",  #  This is role is used in the SCOTUS corpus. It also includes the role of Judge.
            "Attorney": "NA",  # This is the general role for attorneys, lawyers, prosecutors, etc.
            "Doctor": "NA",  # This is the general role for doctors.
            "Nurse": "NA",  # This is the general role for nurses.
            "Student": "NA",  #  Specific forms of this general role include Graduate Student, Senior, High_Schooler, and so on.
            "Teacher": "NA",  # This is the general role for Teachers. Specific forms of this general role include Instuctor, Advisor, Faculty, Professor, Tutor, or T_A.
            "Host": "NA",  #  Specific forms of this general role include ShowHost, Interviewer, and CallTaker.
            "Guest": "NA",  # Specific forms of this general role include ShowGuest, Interviewee, and Caller.
            "Leader": "NA",  # Specific forms of this general role include Group_Leader, Panel_Moderator, Committee_Chair, Facilitator, Tour_Guide, Tour_Leader, Peer_Leader, Chair, or Discussion_Leader.
            "Member": "NA",  # Specific forms of this general role include Committee_Member, Group_Member, Panelist, and Tour_Participant.
            "Narrator": "NA",  # This is a role for presentations of stories.
            "Speaker": "NA",  # Specific forms of this general role include Lecturer, Presenter, Introducer, Welcomer, and Main_Speaker.
            "Audience": "NA",  # This is the general role for single audience members.
        },
    )

    ADDRESSEE_TABLE = defaultdict(
        lambda: "NA", {"FEM": "A", "MAL": "A", "OCH": "C", "CHI": "T"}
    )

    @staticmethod
    def role_to_addressee(role):
        return ChatConverter.ADDRESSEE_TABLE[ChatConverter.SPEAKER_ROLE_TO_TYPE[role]]

    @staticmethod
    def convert(filename: str, filter=None, **kwargs) -> pd.DataFrame:

        import pylangacq

        reader = pylangacq.Reader.from_files([filename])
        participants = reader.headers()[0]["Participants"]
        roles = defaultdict(
            lambda: "NA",
            {p: participants[p]["role"] for p in reader.headers()[0]["Participants"]},
        )

        df = pd.DataFrame(reader.utterances())

        #no segments in the file
        if not df.shape[0]: return pd.DataFrame()
        ### extract tiers
        df["transcription"] = df.apply(
            lambda r: r["tiers"][r["participant"]], axis=1
        ).str.replace(
            r"([\x00-\x1f\x7f-\x9f]+[0-9]+\_[0-9]+[\x00-\x1f\x7f-\x9f])$",
            "",
            regex=True,
        )

        df["tiers"] = df["tiers"].apply(
            lambda d: {k.replace("%", ""): d[k] for k in d.keys() if k[0] == "%"}
        )
        df = pd.concat(
            [df.drop(["tiers"], axis=1), df["tiers"].apply(lambda x: pd.Series(x) if x else pd.Series(dtype='object'))], axis=1
        )

        df["segment_onset"] = df["time_marks"].apply(lambda tm: tm[0] if tm else "NA")
        df["segment_offset"] = df["time_marks"].apply(lambda tm: tm[1] if tm else "NA")

        df["speaker_id"] = df["participant"]
        df["speaker_role"] = df["participant"].replace(roles)
        df["speaker_type"] = df["speaker_role"].map(ChatConverter.SPEAKER_ROLE_TO_TYPE)

        df["words"] = df["tokens"].apply(
            lambda l: len(
                [t["word"] for t in l if re.search(r'[^\W\d_]', t["word"], re.UNICODE)]
            )
        )

        if "add" in df.columns:
            df["addressee"] = (
                df["speaker_type"]
                .fillna("")
                .replace({"NA": ""})
                .apply(
                    lambda s: ",".join(
                        sorted(
                            [
                                ChatConverter.role_to_addressee(roles[x.strip()])
                                for x in str(s).split(",")
                            ]
                        )
                    )
                )
            )

        initial_size = df.shape[0]
        df = df[(df["segment_onset"] != "NA") & (df["segment_offset"] != "NA")]
        if df.shape[0] < initial_size : print("WARNING : Some annotations in file '{}' don't have timestamps, the importation will discard those lines".format(filename))
        df.drop(columns=["participant", "tokens", "time_marks"], inplace=True)
        df.fillna("NA", inplace=True)

        return df
