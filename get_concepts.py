from pymetamap import MetaMap
import json
from nltk.tokenize import sent_tokenize
from tqdm import tqdm


def process_concepts(entities):
    results = []
    for entity in entities:
        item = dict()
        item['semtypes'] = entity.semtypes.lstrip('[').rstrip(']').split(',')
        # define filter list
        if 'qnco' in item['semtypes']:  # filter out Quantitative Concept: per year
            continue
        if 'tmco' in item['semtypes']:  # filter out Temporal Concept: day
            continue
        if 'hlca' in item['semtypes']:  # Health Care Activity: Hospital admission; Tapering - action;
            continue
        if 'idcn' in item['semtypes']:  # Idea or Concept: Presentation
            continue
        if 'hcro' in item['semtypes']:  # Health Care Related Organization: Accident and Emergency department
            continue
        if 'clna' in item['semtypes']:  # Clinical Attribute: History of present illness
            continue
        if 'ftcn' in item['semtypes']:  # Functional Concept: Negation, Extraocular; Due to
            continue
        if 'qlco' in item['semtypes']:  # Qualitative Concept: Started
            continue
        if 'fndg' in item['semtypes']:  # Finding: Present
            continue
        if 'acty' in item['semtypes']:  # Activity: Obscure; Assessed; Departure - action
            continue
        if 'mnob' in item['semtypes']:  # Manufactured Object: Machine; Beds
            continue
        if 'plnt' in item['semtypes']:  # Plant
            continue
        if 'podg' in item['semtypes']:  # Patient or Disabled Group: Patients
            continue
        if 'popg' in item['semtypes']:  # Population Group
            continue
        if 'prog' in item['semtypes']:  # Professional or Occupational Group: Physicians
            continue
        if 'pros' in item['semtypes']:  # Professional Society
            continue
        if 'elii' in item['semtypes']:  # Element, Ion, or Isotope: lead
            continue
        if 'anim' in item['semtypes']:  # Animal: Show
            continue
        if 'inpr' in item['semtypes']:  # Intellectual Product: Code; Telephone Number
            continue
        if 'orgf' in item['semtypes']:  # Organism Function: Movement; Expiration, function; Inspiration function
            continue
        if 'npop' in item['semtypes']:  # Natural Phenomenon or Process: Saturated
            continue
        if 'lang' in item['semtypes']:  # Language: Herero language
            continue
        if 'spco' in item['semtypes']:  # Spatial Concept: Scattered; Round shape
            continue
        if 'bpoc' in item['semtypes']:  # Body Part, Organ, or Organ Component: Eminence
            continue

        item['index'] = len(results)
        item['mm'] = entity.mm
        item['score'] = entity.score
        item['preferred_name'] = entity.preferred_name
        item['cui'] = entity.cui
        item['trigger'] = entity.trigger.lstrip('[').rstrip(']').split(',')
        item['pos_info'] = entity.pos_info
        results.append(item)

    return results


def extract_entities(docs):
    results = []
    for doc in docs:
        mm = MetaMap.get_instance('/data/xiaolei/public_mm/bin/metamap')
        # parameters documentation: https://metamap.nlm.nih.gov/Docs/MM_2016_Usage.pdf
        # https://metamap.nlm.nih.gov/Docs/README_javaapi.shtml
        concepts = mm.extract_concepts(
            [doc], word_sense_disambiguation=True, unique_acronym_variants=True,
            ignore_stop_phrases=True, no_derivational_variants=True, no_nums=['all'],
            exclude_sts=[
                'bpoc', 'spco', 'lang', 'npop', 'orgf', 'qnco', 'tmco', 'hlca', 'idcn', 'hcro', 'clna',
                'ftcn', 'qlco', 'fndg', 'acty', 'mnob', 'plnt', 'podg', 'popg', 'prog', 'pros', 'elii',
                'anim',
            ],
        )
        results.append(concepts)

    # below implementation will cause insuficient memory error, even with 16G XMx
    # results = [[]] * len(docs)
    # concepts = mm.extract_concepts(docs, ids=list(range(len(docs))), word_sense_disambiguation=True)
    # for concept in concepts:
    #     results[int(concept.index)].append(concept)

    results = [process_concepts(item) for item in results]
    return results


def process_json(file_path, dname):
    entity_details = open('./resources/entity_details_{}.txt'.format(dname), 'w')
    with open(file_path + '_bk', 'w') as wfile:
        with open(file_path) as dfile:
            for idx, line in tqdm(enumerate(dfile)):
                user = json.loads(line)
                documents = [item['text'] for item in user['docs']]
                results = extract_entities(documents)
                for jdx, concepts in enumerate(results):
                    user['docs'][jdx]['concepts'] = concepts

                    # record all concepts in a separate file as well
                    for cpt in concepts:
                        cpt['uid'] = user['uid']
                        entity_details.write(json.dumps(cpt) + '\n')

                wfile.write(json.dumps(user) + '\n')

    # replace with new data file
    # os.remove(file_path)
    # os.rename(file_path + '_bk', file_path)


if __name__ == '__main__':
    dlist = ['diabetes', 'mimic-iii']  # 'diabetes', 'mimic-iii'
    dir_pattern = './data/processed_data/{}/{}.json'

    for task in dlist:
        task_path = dir_pattern.format(task, task)

        process_json(task_path, task)
