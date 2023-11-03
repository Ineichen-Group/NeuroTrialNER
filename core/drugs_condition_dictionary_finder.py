import csv
import re
import pathlib
import pandas as pd
this_path = pathlib.Path(__file__).parent.resolve()

drug_variant_to_canonical = {}
drug_canonical_to_data = {}
condition_variant_to_canonical = {}

exclusions = {'ACACIA',
              'ACETAMIDE',
              'ACETATE',
              'ACETOPHENONE',
              'ACETYLCHOLINE',
              'ACETYLENE',
              'ACRIFLAVINE',
              'ACT',
              'ACTINIUM',
              'ADENINE',
              'ADENOSINE',
              'ADRENALONE',
              'AGMATINE',
              'AIM',
              'ALANINE',
              'ALLANTOIN',
              'ALLSPICE',
              'ALMOND',
              'ALUMINIUM',
              'AMBER',
              'AMEN',
              'AMMONIA',
              'AMYLAMINE',
              'AMYLOPECTIN',
              'AMYLOSE',
              'ANETHOLE',
              'ANILINE',
              'ANTIMONY',
              'ANTIPYRINE',
              'APIGENIN',
              'APPLE',
              'APRICOT',
              'ARGININE',
              'ARTICHOKE',
              'ASPARAGINE',
              'ASPARAGUS',
              'AVOCADO',
              'BA',
              'BANANA',
              'BARIUM',
              'BARLEY',
              'BASIL',
              'BEAN',
              'BEEF',
              'BEESWAX',
              'BEET',
              'BELLADONNA',
              'BENTONITE',
              'BENZIMIDAZOLE',
              'BENZOIN',
              'BENZOPHENONE',
              'BENZYLAMINE',
              'BERBERINE',
              'BERKELIUM',
              'BETAINE',
              'BILBERRY',
              'BLACKBERRY',
              'BLUEBERRY',
              'BLUEFISH',
              'BORNEOL',
              'BORON',
              'BROCCOLI',
              'BROMOFORM',
              'BUCKWHEAT',
              'BUTYLAMINE',
              'CABALETTA',
              'CABBAGE',
              'CADAVERINE',
              'CADMIUM',
              'CAFFEINE',
              'CALCIUM',
              'CAMPHANE',
              'CAMPHENE',
              'CAMPHOR',
              'CANTALOUPE',
              'CAPSAICIN',
              'CAPSICUM',
              'CARAWAY',
              'CARBAZOLE',
              'CARBONATE',
              'CARDAMOM',
              'CARNOSINE',
              'CAROB',
              'CARROT',
              'CARVACROL',
              'CASEIN',
              'CASHEW',
              'CATFISH',
              'CAULIFLOWER',
              'CELERY',
              'CELLOBIOSE',
              'CESIUM',
              'CHERRY',
              'CHICKEN',
              'CHLORINE',
              'CHOLECYSTOKININ',
              'CHOLESTEROL',
              'CHOLINE',
              'CHROMIUM',
              'CHRYSIN',
              'CHYMOTRYPSIN',
              'CINCHOPHEN',
              'CINNAMALDEHYDE',
              'CINNAMON',
              'CLOVE',
              'COBALT',
              'COCAINE',
              'COCARBOXYLASE',
              'COCOA',
              'COCONUT',
              'COPPER',
              'CORN',
              'CORTICOSTERONE',
              'COTTON',
              'COUMARIN',
              'CRANBERRY',
              'CREATINE',
              'CREATININE',
              'CRESOL',
              'CREST',
              'CROCIN',
              'CROTONALDEHYDE',
              'CUCUMBER',
              'CUMIDINE',
              'CUMIN',
              'CURCUMIN',
              'CYANAMIDE',
              'CYCLOHEXANOL',
              'CYCLOHEXANONE',
              'CYCLOPROPANE',
              'CYSTEINE',
              'CYSTINE',
              'CYTISINE',
              'DATE',
              'DEXTRAN',
              'DIAMORPHINE',
              'DICHROMATE',
              'DIETHYLSTILBESTROL',
              'DIGITOXIN',
              'DIHYDROTACHYSTEROL',
              'DILL',
              'DINITROPHENOL',
              'DIOSMIN',
              'DIPHENYLGUANIDINE',
              'DUCK',
              'DUROQUINONE',
              'ECGONINE',
              'ECHINACEA',
              'EGG',
              'EGGPLANT',
              'ELM',
              'EMETINE',
              'EMODIN',
              'ENDURA',
              'EOSIN',
              'EPHEDRINE',
              'ERGOMETRINE',
              'ERGOSTEROL',
              'ERGOTAMINE',
              'ERYTHRITOL',
              'ESCULIN',
              'ESTRADIOL',
              'ESTRIOL',
              'ESTRONE',
              'ESTROGEN',
              'ETHANOL',
              'ETHANOLAMINE',
              'ETHER',
              'EUCALYPTOL',
              'EUGENOL',
              'FARNESOL',
              'FENCHONE',
              'FENNEL',
              'FIBRIN',
              'FIG',
              'FISETIN',
              'FLAVONE',
              'FLEET',
              'FLOUNDER',
              'FLUORESCEIN',
              'FLUORESCIN',
              'FLUORIDE',
              'FLUORSPAR',
              'FORMALDEHYDE',
              'FRANKINCENSE',
              'FRUCTOSE',
              'FUCOSE',
              'FUCOXANTHIN',
              'GADOLINIUM',
              'GALACTOSE',
              'GARLIC',
              'GELATIN',
              'GENISTEIN',
              'GERANIOL',
              'GINGER',
              'GINSENG',
              'GLUCOSAMINE',
              'GLUTATHIONE',
              'GLYCERIN',
              'GLYCINE',
              'GLYCOLIDE',
              'GOLD',
              'GOLDENSEAL',
              'GOOSE',
              'GOSSYPOL',
              'GRAPE',
              'GRAPEFRUIT',
              'GUAIACOL',
              'GUANIDINE',
              'GUANINE',
              'GUANOSINE',
              'GUVACINE',
              'HADDOCK',
              'HARMALINE',
              'HARMINE',
              'HAZELNUT',
              'HELIUM',
              'HEMATIN',
              'HEMIN',
              'HEMOGLOBIN',
              'HEPARIN',
              'HERRING',
              'HESPERIDIN',
              'HEXESTROL',
              'HISTAMINE',
              'HISTIDINE',
              'HONEY',
              'HYALURONIDASE',
              'HYDROGEN',
              'HYDROTALCITE',
              'HYOSCYAMINE',
              'HYPERICIN',
              'HYPOCHLORITE',
              'HYPOPHOSPHITE',
              'HYPOXANTHINE',
              'IMIDAZOLE',
              'INDIRUBIN',
              'INDIUM',
              'INDOLE',
              'INOSITOL',
              'INULIN',
              'IODIDE',
              'IODINE',
              'IODOBENZENE',
              'IODOFORM',
              'IPECAC',
              'IRON',
              'ISATIN',
              'ISOEUGENOL',
              'ISOFLAVONE',
              'ISOLEUCINE',
              'ISOPENTANE',
              'ISOQUERCITRIN',
              'ISOQUINOLINE',
              'KALE',
              'KAOLIN',
              'KAVA',
              'LACTOSE',
              'LAMB',
              'LANOLIN',
              'LANTHANUM',
              'LECITHIN',
              'LEEK',
              'LEMON',
              'LENTIL',
              'LETTUCE',
              'LEUCINE',
              'LICORICE',
              'LINDANE',
              'LITHIUM',
              'LOBELINE',
              'LOBSTER',
              'LUPEOL',
              'LUTEIN',
              'LUTEOLIN',
              'LYCOPENE',
              'LYSINE',
              'LYSOZYME',
              'MACKEREL',
              'MAGNESIUM',
              'MALTODEXTRIN',
              'MALTOSE',
              'MANGANESE',
              'MANGO',
              'MANNITOL',
              'MANNOSE',
              'MENADIONE',
              'MENTHOL',
              'MENTHONE',
              'METHANE',
              'METHIONINE',
              'METHYLAMINE',
              'MOLYBDATE',
              'MOLYBDENUM',
              'MORPHOLINE',
              'MUSKMELON',
              'MYRICETIN',
              'MYRRH',
              'NARINGENIN',
              'NECTARINE',
              'NEODYMIUM',
              'NEON',
              'NIACIN',
              'NICOTINAMIDE',
              'NICOTINE',
              'NIKETHAMIDE',
              'NIOBIUM',
              'NITRATE',
              'NITRITE',
              'NITROGEN',
              'NITROGLYCERIN',
              'NITROPRUSSIDE',
              'NORLEUCINE',
              'NUTMEG',
              'OAT',
              'OKRA',
              'OLEANDRIN',
              'ONION',
              'OPIUM',
              'ORANGE',
              'ORNITHINE',
              'ORRIS',
              'OUABAIN',
              'OXYGEN',
              'OXYTOCIN',
              'OYSTER',
              'OZONE',
              'PAPAIN',
              'PAPAVERINE',
              'PAPAYA',
              'PAPRIKA',
              'PARALDEHYDE',
              'PARSLEY',
              'PARSNIP',
              'PATROL',
              'PEA',
              'PEACH',
              'PEANUT',
              'PEAR',
              'PECAN',
              'PECTIN',
              'PENTAERYTHRITOL',
              'PEPPERMINT',
              'PEPSIN',
              'PERCH',
              'PERCHLORATE',
              'PETROLATUM',
              'PHENACETIN',
              'PHENOL',
              'PHENOLPHTHALEIN',
              'PHENOTHIAZINE',
              'PHENYLACETALDEHYDE',
              'PHENYLALANINE',
              'PHOSPHOCREATINE',
              'PHOSPHORUS',
              'PHTHALOCYANINE',
              'PHYSOSTIGMINE',
              'PINEAPPLE',
              'PINITOL',
              'PIPERAZINE',
              'PIPERINE',
              'PISTACHIO',
              'PLATINUM',
              'PLUM',
              'POMEGRANATE',
              'PORK',
              'POTASSIUM',
              'POTATO',
              'POULTRY',
              'PROCAINE',
              'PROFLAVINE',
              'PROGESTERONE',
              'PROLINE',
              'PROTAMINE',
              'PROTHROMBIN',
              'PROTOCATECHUALDEHYDE',
              'PROTOPORPHYRIN',
              'PSEUDOEPHEDRINE',
              'PSEUDOTROPINE',
              'PUMICE',
              'PUMPKIN',
              'PURSLANE',
              'PUTRESCINE',
              'RABBIT',
              'RADISH',
              'RASPBERRY',
              'REPOSAL',
              'RESORCINOL',
              'RHAMNOSE',
              'RHEIN',
              'RHUBARB',
              'RIBOFLAVIN',
              'RIBOSE',
              'RICE',
              'ROSEMARY',
              'ROSIN',
              'ROTENONE',
              'RUBIDIUM',
              'RUTIN',
              'RYE',
              'SACCHARIN',
              'SAFFLOWER',
              'SAGE',
              'SALICYLAMIDE',
              'SALOL',
              'SAMARIUM',
              'SARCOSINE',
              'SCALLOP',
              'SCOPOLAMINE',
              'SELENIUM',
              'SENNA',
              'SERINE',
              'SHRIMP',
              'SILICON',
              'SILVER',
              'SMELT',
              'SNAIL',
              'SONATA',
              'SORBITOL',
              'SOYBEAN',
              'SPARTEINE',
              'SPEARMINT',
              'SPERMACETI',
              'SPERMIDINE',
              'SPERMINE',
              'SPHINGOSINE',
              'SPINACH',
              'SQUALENE',
              'SQUASH',
              'STRAWBERRY',
              'SUCCINIMIDE',
              'SUCROSE',
              'SWORDFISH',
              'TABLOID',
              'TAGATOSE',
              'TALC',
              'TANGERINE',
              'TANTALUM',
              'TARTRONATE',
              'TAURINE',
              'TENUATE',
              'TERPINEOL',
              'TESTOSTERONE',
              'TETRAMETHYLAMMONIUM',
              'THEOBROMINE',
              'THEOPHYLLINE',
              'THIAMINE',
              'THREONINE',
              'THROMBIN',
              'THYME',
              'THYMINE',
              'THYMOL',
              'THYROID',
              'TING',
              'TITANIUM',
              'TOCOPHEROL',
              'TOLUENE',
              'TOMATO',
              'TRAGACANTH',
              'TREHALOSE',
              'TRIBUTYRIN',
              'TRIOLEIN',
              'TROUT',
              'TRYPSIN',
              'TUBOCURARINE',
              'TUNA',
              'TURKEY',
              'TURMERIC',
              'TURNIP',
              'TURPENTINE',
              'TYRAMINE',
              'TYROSINASE',
              'TYROSINE',
              'URACIL',
              'UREA',
              'URETHANE',
              'VALERIAN',
              'VALINE',
              'VANADIUM',
              'VANILLA',
              'VEAL',
              'VENISON',
              'VERBENONE',
              'VERSED',
              'VITAMIN',
              'VORTEX',
              'WATER',
              'WATERMELON',
              'WHEAT',
              'WORMWOOD',
              'XANTHINE',
              'XENON',
              'XYLITOL',
              'XYLOSE',
              'YEAST',
              'ZINC',
              'ZINGERONE'}

# Drug names which are sufficiently generic that they may occur lower case
words_to_allow_lower_case = {'amphetamine',
                             'andrographolide',
                             'apomorphine',
                             'arbutin',
                             'arecoline',
                             'aspirin',
                             'atropine',
                             'bacitracin',
                             'barbital',
                             'benzocaine',
                             'benzofuran',
                             'benzylpenicillin',
                             'biguanide',
                             'biotin',
                             'cannabinol',
                             'cantharidin',
                             'carbromal',
                             'cathine',
                             'chloramphenicol',
                             'chloroform',
                             'chloroquine',
                             'codeine',
                             'colchicine',
                             'cortisone',
                             'emend',
                             'epinephrine',
                             'estrogen',
                             'ethylenediamine',
                             'ethylmorphine',
                             'factive',
                             'fibrinolysin',
                             'hexylresorcinol',
                             'hydroquinine',
                             'hydroquinone',
                             'lustral',
                             'methadone',
                             'methenamine',
                             'morphine',
                             'oxyquinoline',
                             'paregoric',
                             'penicillin',
                             'pentobarbital',
                             'phenobarbital',
                             'picropodophyllin',
                             'picrotoxin',
                             'pilocarpine',
                             'podophyllin',
                             'psyllium',
                             'pyrazole',
                             'pyridoxine',
                             'pyruvaldehyde',
                             'quercetin',
                             'quinacrine',
                             'quinidine',
                             'quinine',
                             'streptomycin',
                             'strychnine',
                             'sulfadiazine',
                             'sulfaguanidine',
                             'sulfamerazine',
                             'sulfamethazine',
                             'sulfamethylthiazole',
                             'sulfanilamide',
                             'sulfapyridine',
                             'sulfaquinoxaline',
                             'sulfathiazole',
                             'thymoquinone',
                             'thyroglobulin',
                             'trichloroethylene',
                             'trinitrotoluene',
                             'tryptophan',
                             'yohimbine'}
# Allows for detection of
# Aluminum Hydroxide,
# Alu-Tab,
# AlternaGEL,
# Interferon beta-1a
# 3,4-Diaminopyridine
# Excludes names like: "Aluminum Hydroxide and Magnesium Hydroxide"
# TODO: Issue with differently capitalized drugs, e.g. 3,4-D(d)iaminopyridine
variant_regex = re.compile(r'^[A-Za-z0-9,]+[ -]?[A-Za-z0-9\-]+(?:[ -][A-Z])?$')


def add_variant(canonical_name, variant, dict_type="drug"):
    if dict_type == "drug":
        if variant not in drug_variant_to_canonical:
            drug_variant_to_canonical[variant] = set()
        drug_variant_to_canonical[variant].add(canonical_name)
    else:
        if variant not in condition_variant_to_canonical:
            condition_variant_to_canonical[variant] = set()
        condition_variant_to_canonical[variant].add(canonical_name)


def generate_conditions_lookup_dictionary(df):
    synonyms_dict = {}

    for index, row in df.iterrows():
        icd_title = row['ICD Title']
        mesh_name = row['MeSH Common name']
        if pd.notna(row['MeSH Synonyms']):
            synonyms_list = row['MeSH Synonyms'].split('|')
            for synonym in synonyms_list:
                synonym = synonym.strip().lower()
                mesh_name = mesh_name.lower()
                synonyms_dict = add_variant(mesh_name, synonym, synonyms_dict)
        elif pd.notna(row['ICD Title']):
            icd_title = icd_title.lower()
            synonyms_dict = add_variant(icd_title, icd_title, synonyms_dict)
        elif pd.notna(row['MeSH Common name']):
            mesh_nam = mesh_name.lower()
            synonyms_dict = add_variant(mesh_name, mesh_name, synonyms_dict)

    return synonyms_dict


def add_drug(id, synonyms):
    synonyms = [s.strip() for s in synonyms]

    #TODO: add using an exclusion list as a parameter option to the function
    #if re.sub("[- ].+", "", synonyms[0].upper()) in exclusions:
    #    return
    if not variant_regex.match(synonyms[0]):
        return
    if synonyms[0] not in drug_canonical_to_data:
        drug_canonical_to_data[synonyms[0]] = {"name": synonyms[0], "synonyms": set()}
    if id.startswith("a"):
        drug_canonical_to_data[synonyms[0]]["medline_plus_id"] = id
    elif id.startswith("https://www.nhs.uk"):
        drug_canonical_to_data[synonyms[0]]["nhs_url"] = id
    elif id.startswith("https://en.wikipedia"):
        drug_canonical_to_data[synonyms[0]]["wikipedia_url"] = id
    elif id.startswith("DB"):
        drug_canonical_to_data[synonyms[0]]["drugbank_id"] = id
    else:
        drug_canonical_to_data[synonyms[0]]["mesh_id"] = id
    for variant in synonyms:
        #if re.sub(" .+", "", variant.upper()) in exclusions:
        #    return
        if variant_regex.match(variant):
            drug_canonical_to_data[synonyms[0]]["synonyms"].add(variant)
            add_variant(synonyms[0], variant)
            add_variant(synonyms[0], variant.upper())
            if variant.lower() in words_to_allow_lower_case:
                add_variant(synonyms[0], variant.lower())


with open(this_path.joinpath("./data/drugs_dictionary_medlineplus.csv"), 'r', encoding="utf-8") as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    headers = None
    for row in spamreader:
        if not headers:
            headers = row
            continue
        id = row[0]
        name = row[1]
        #if id == "a699048":
        #    print("check")
        synonyms = row[2].split(r"|")

        name = re.sub(
            " (Injection|Oral Inhalation|Transdermal|Ophthalmic|Topical|Vaginal Cream|Nasal Spray|Transdermal Patch|Rectal)",
            "", name)

        add_drug(id, [name] + synonyms)

with open(this_path.joinpath("./data/drugs_dictionary_nhs.csv"), 'r', encoding="utf-8") as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    headers = None
    for row in spamreader:
        if not headers:
            headers = row
            continue
        id = row[0]
        name = row[1]
        synonyms = row[2].split(r"|")

        add_drug(id, [name] + synonyms)


with open(this_path.joinpath("./data/drugs_dictionary_wikipedia.csv"), 'r', encoding="utf-8") as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    headers = None
    for row in spamreader:
        if not headers:
            headers = row
            continue
        id = row[0]
        name = row[1]
        synonyms = row[2].split(r"|")

        add_drug(id, [name] + synonyms)
        
with open(this_path.joinpath("./data/drugs_dictionary_mesh.csv"), 'r', encoding="utf-8") as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    headers = None
    for row in spamreader:
        if not headers:
            headers = row
            continue
        id = row[0]
        name = row[1]
        synonyms = row[2].split(r"\|")
        add_drug(id, [name] + synonyms)

# adding for the full db with product names included as synonyms
# TODO: create a parametrized function from this, not hard-coded inline as it is
is_new_format = False
if is_new_format:
    with open(this_path.joinpath("./data/drugdb_full_database_parsed.csv"), 'r', encoding="utf-8") as csvfile:
        spamreader = csv.reader(csvfile, delimiter=';')
        headers = None
        for row in spamreader:
            #print(row)
            if not headers:
                headers = row
                continue
            id = row[0]
            name = row[1]
            synonyms = row[4].split(r"|")
            products = row[5].split(r"|")
            syn_prod = synonyms + products
            add_drug(id, [name] + syn_prod)
# no product names considered
else:
    with open(this_path.joinpath("./data/drugbank vocabulary.csv"), 'r', encoding="utf-8") as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        headers = None
        for row in spamreader:
            if not headers:
                headers = row
                continue
            id = row[0]
            name = row[2]
            synonyms = row[5].split(r"|")
            add_drug(id, [name] + synonyms)
        


def find_drugs(tokens: list, is_ignore_case: bool = False):
    drug_matches = []
    is_exclude = set()
    # Search for 2 token sequences
    for token_idx, token in enumerate(tokens[:-1]):
        cand = token + " " + tokens[token_idx + 1]
        if is_ignore_case:
            match = drug_variant_to_canonical.get(cand.upper(), None)
        else:
            match = drug_variant_to_canonical.get(cand, None)
        if match:
            for m in match:
                drug_matches.append((drug_canonical_to_data[m], token_idx, token_idx + 1))
                is_exclude.add(token_idx)
                is_exclude.add(token_idx + 1)

    for token_idx, token in enumerate(tokens):
        if token_idx in is_exclude:
            continue
        if is_ignore_case:
            match = drug_variant_to_canonical.get(token.upper(), None)
        else:
            match = drug_variant_to_canonical.get(token, None)
        if match:
            for m in match:
                drug_matches.append((drug_canonical_to_data[m], token_idx, token_idx))

    return drug_matches

def find_drugs_normalized_output(sentence: str, tokens: list, is_ignore_case: bool = False, return_word_not_label: bool = False):
    # this code add additional model_annotations with the start and end character of the found drugs
    # needed training outputs for SpaCy NER
    drug_matches = []
    is_exclude = set()
    all_char_indices = []

    # Search for 2 token sequences
    for token_idx, token in enumerate(tokens[:-1]):
        cand = token + " " + tokens[token_idx + 1]
        if is_ignore_case:
            match = drug_variant_to_canonical.get(cand.upper(), None)
        else:
            match = drug_variant_to_canonical.get(cand, None)
        if match and cand.upper() != "MG": # issue with 10 mg -> mg considered a DRUG
            if return_word_not_label:
                char_indices = [(i.start(), i.end()) + (cand.lower(),) for i in re.finditer(cand.upper(), sentence.upper())]
            else:
                char_indices = [(i.start(), i.end()) + ("DRUG",) for i in re.finditer(cand.upper(), sentence.upper())]
            all_char_indices = all_char_indices + char_indices
            for m in match:
                drug_matches.append((drug_canonical_to_data[m], token_idx, token_idx + 1))
                is_exclude.add(token_idx)
                is_exclude.add(token_idx + 1)

    for token_idx, token in enumerate(tokens):
        if token_idx in is_exclude:
            continue
        if is_ignore_case:
            match = drug_variant_to_canonical.get(token.upper(), None)
        else:
            match = drug_variant_to_canonical.get(token, None)
        if match and token.upper() != "MG":
            if return_word_not_label:
                char_indices = [(i.start(), i.end()) + (token.lower(),) for i in re.finditer(token.upper(), sentence.upper())]
            else:
                char_indices = [(i.start(), i.end()) + ("DRUG",) + (token.lower(),) for i in re.finditer(token.upper(), sentence.upper())]
            all_char_indices = all_char_indices + char_indices
            for m in match:
                drug_matches.append((drug_canonical_to_data[m], token_idx, token_idx))

    all_char_indices = list(set(all_char_indices))
    return drug_matches, all_char_indices #{"entites":all_char_indices}

def load_conditions_dict():
    conditions_db = pd.read_csv("/Users/donevas/Desktop/Projects/Univeristy/PhD/Code/pythonNLP/clinical_trials_ner/entity_linking/data/diseases_dictionary_mesh_icd_20230731.csv")
    synonyms_dict = generate_conditions_lookup_dictionary(conditions_db)

def find_match(token, is_ignore_case: bool = False):
    if is_ignore_case:
        match_drug = drug_variant_to_canonical.get(token.upper(), None)
        match_condition = condition_variant_to_canonical.get(token.lower(), None)
    else:
        match_drug = drug_variant_to_canonical.get(token, None)
        match_condition = condition_variant_to_canonical.get(token, None)
    return match_drug, match_condition

def find_drugs_and_conditions_normalized_BIO_output(tokens: list, is_ignore_case: bool = True):
    load_conditions_dict()
    bio_output = []
    tokens = eval(tokens)
    to_skip = False
    for token_idx, token in enumerate(tokens):
        if to_skip:
            to_skip = False
            continue
        match_drug, match_condition = find_match(token, is_ignore_case)

        if match_drug and token.upper() != "MG":
            bio_output.append("B-DRUG")
        elif match_condition:
            bio_output.append("B-COND")
        else:
            # try if two tokens work
            if token_idx + 1 == len(tokens):
                bio_output.append("O")
                continue
            else:
                cand = token + " " + tokens[token_idx + 1]
                match_drug, match_condition = find_match(cand, is_ignore_case)
                if match_drug:
                    bio_output.append("B-DRUG")
                    bio_output.append("I-DRUG")
                    to_skip = True
                elif match_condition:
                    bio_output.append("B-COND")
                    bio_output.append("I-COND")
                    to_skip = True
                else:
                    bio_output.append("O")

    return bio_output  # {"entites":all_char_indices}


# for local testing
if __name__ == '__main__':
    sentence = "Pilot Test of Flavoxate and Flavoxatum in Multiple Sclerosis: Safety and Tolerability."
    print(find_drugs_normalized_output(sentence, sentence.split(" "), is_ignore_case=True))