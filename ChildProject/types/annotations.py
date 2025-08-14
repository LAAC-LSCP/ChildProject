from enum import Enum


class LENABlockType(Enum):
    """
    In .its files segments are grouped into larger, non-overlapping `vocalisation activity blocks`
    There are two types of blocks; conversations and pauses.
    Conversations have types dependent on who initiates the conversation and who participates in it
    """
    PAUSE = "pause"

    KEY_CHILD_MONOLOGUE = "CM"
    KEY_CHILD_WITH_ADULT = "CIC"
    KEY_CHILD_WITH_OTHER_CHILD = "CIOCX"
    KEY_CHILD_WITH_ADULT_AND_OTHER_CHILD = "CIOCAX"

    FEMALE_ADULT_MONOLOGUE = "AMF"
    FEMALE_ADULT_WITH_KEY_CHILD = "AICF"
    FEMALE_ADULT_WITH_OTHER_CHILD = "AIOCF"
    FEMALE_ADULT_WITH_KEY_CHILD_AND_OTHER_CHILD = "AIOCCXF"

    MALE_ADULT_MONOLOGUE = "AMM"
    MALE_ADULT_WITH_KEY_CHILD = "AICM"
    MALE_ADULT_WITH_OTHER_CHILD = "AIOCM"
    MALE_ADULT_WITH_KEY_CHILD_AND_OTHER_CHILD = "AIOCCXM"

    OTHER_CHILD_MONOLOGUE = "XM"
    OTHER_CHILD_WITH_KEY_CHILD = "XIOCC"
    OTHER_CHILD_WITH_ADULT = "XIOCA"
    OTHER_CHILD_WITH_KEY_CHILD_AND_ADULT_TURNS = "XIC"
    OTHER_CHILD_WITH_KEY_CHILD_AND_ADULT_NO_TURNS = "XIOCAC"


class LENAConversationalFloorType(Enum):
    """
    Some segments within `vocalisation activity blocks` have special functions and are labelled accordingly

    We define
    - floor initialization: the speaker speaks for the first time in this block
    - floor holding: the speaker has spoken before in this block
    """
    FLOOR_INITIALIZATION = "FI"
    FLOOR_HOLDING = "FH"


class LENAConversationalStatus(Enum):
    """
    Running status of the segment in the conversation block
    """
    BEGIN = "BC"
    RUNNING = "RC"
    END_OF_BLOCK = "EC"


class LENASpeakerType(Enum):
    """
    Every segment is labelled according to the predicted source, which we dub, inexactly, "speaker type"
    """
    MALE_ADULT = "MAN"
    MALE_ADULT_FAINT = "MAF"

    FEMALE_ADULT = "FAN"
    FEMALE_ADULT_FAINT = "FAF"

    KEY_CHILD = "CHN"
    KEY_CHILD_FAINT = "CHF"

    OTHER_CHILD = "CXN"
    OTHER_CHILD_FAINT = "CXF"

    NOISE = "NON"
    NOISE_FAINT = "NOF"

    OVERLAPPING_NOISE = "OLN"
    OVERLAPPING_NOISE_FAINT = "OLF"

    TV_OR_ELECTRONIC_SOUNDS = "TVN"
    TV_OR_ELECTRONIC_SOUNDS_FAINT = "TVF"

    SILENCE = "SIL"
    UNCERTAIN_OR_FAINT = "FUZ"


class LENAConversationalTurnType(Enum):
    """
    When the speaker switches from one segment to another, this will under certain conditions
    be counted as a conversational turn
    """
    TURN_INITIALIZATION_WITH_FEMALE = "TIFI"
    TURN_INITIALIZATION_WITH_MALE = "TIMI"

    TURN_RESPONSE_WITH_FEMALE = "TIFR"
    TURN_RESPONSE_WITH_MALE = "TIMR"
    
    TURN_END_WITH_FEMALE = "TIFE"
    TURN_END_WITH_MALE = "TIME"
    
    OTHER_CHI_WITH_KEY_CHI_AND_ADULT = "NT"
