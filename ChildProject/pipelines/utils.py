from pathlib import Path
import os.path
import pympi


def choose_template(template):
    """
    Update: Choose between three template.Basic, native and non-native.
    """
    if template == 'basic':
        return 'ACLEW-basic-template_all-tiers.etf', 'ACLEW-basic-template_all-tiers.pfsx'
    if template == 'native':
        return 'ACLEW-LAAC-native.etf','ACLEW-LAAC-native.pfsx'
    if template == 'non-native':
        return 'ACLEW-LAAC-non-native.etf','ACLEW-LAAC-non-native.pfsx'

def create_eaf(etf_path, id, output_dir, timestamps_list, eaf_type,contxt_on, contxt_off,template):
    
    print("ACLEW ID: ", id)
    eaf = pympi.Elan.Eaf(etf_path)
    ling_type = "transcription"
    eaf.add_tier("code_"+eaf_type, ling=ling_type)
    eaf.add_tier("context_"+eaf_type, ling=ling_type)
    eaf.add_tier("code_num_"+eaf_type, ling=ling_type)
    for i, ts in enumerate(timestamps_list):
        print("Creating eaf code segment # ", i+1)
        print("enumerate makes: ", i, ts)
        whole_region_onset = ts[0]
        whole_region_offset = ts[1]
        #print whole_region_offset, whole_region_onset
        context_onset = int(float(whole_region_onset) - float(contxt_on)*60000)
        #for float / integer unmatch float()
        context_offset = int(float(whole_region_offset) + float(contxt_off)*60000)
        if context_onset < 0:
            context_onset = 0.0
        codeNumVal = eaf_type + str(i+1)
        eaf.add_annotation("code_"+eaf_type, whole_region_onset, whole_region_offset)
        eaf.add_annotation("code_num_"+eaf_type, whole_region_onset, whole_region_offset, value=codeNumVal)
        eaf.add_annotation("context_"+eaf_type, context_onset, context_offset)

    import pdb
    pdb.set_trace()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    eaf.to_file(output_dir / "{}.eaf".format(id))
    for i in eaf.get_tier_names():
        print(i,":",eaf.get_annotation_data_for_tier(i))
    return eaf


