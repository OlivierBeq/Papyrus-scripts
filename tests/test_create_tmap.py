# -*- coding: utf-8 -*-
from collections import Counter
from typing import List, Union

import numpy as np
import pandas as pd
import tmap as tm
from faerun import Faerun
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm


def get_mol_props_for_map(mols: List[Chem.Mol]):
    """Return the heavy atom count, carbon fraction, ring atom fraction and largest ring size."""
    hac = []
    c_frac = []
    ring_atom_frac = []
    largest_ring_size = []

    for mol in mols:
        atoms = mol.GetAtoms()
        size = mol.GetNumHeavyAtoms()
        n_c = 0
        n_ring_atoms = 0
        for atom in atoms:
            if atom.IsInRing():
                n_ring_atoms += 1
            if atom.GetSymbol().lower() == "c":
                n_c += 1

        c_frac.append(n_c / size)
        ring_atom_frac.append(n_ring_atoms / size)
        sssr = AllChem.GetSymmSSSR(mol)
        if len(sssr) > 0:
            largest_ring_size.append(max([len(s) for s in sssr]))
        else:
            largest_ring_size.append(0)
        hac.append(size)
    return hac, c_frac, ring_atom_frac, largest_ring_size


def ordered_top_labels(series: Union[pd.Series, List], top_n: int):
    """Return the Faerun labels and data of the top occurences, others categorized as 'Other'.

    :param series: the data
    :param top_n: the number of categories other than 'Other' to keep
    """
    labels, data = Faerun.create_categories(series)
    top = [i for i, _ in Counter(data).most_common(top_n)]

    top_labels = [(7, "Other")]
    map_ = [7] * len(data)
    value = 1
    for i, name in labels:
        if i in top:
            v = value
            if v == 7:
                v = 0
            top_labels.append((v, name))
            map_[i] = v
            value += 1
    data = [map_[val] for _, val in enumerate(data)]
    return labels, data


if __name__ == '__main__':

    lf = tm.LSHForest(7437, 1024, 32)
    hac, c_frac, ring_atom_frac, ring_size = [], [], [], []
    organisms, lengths, l1, l2, l3, l4, l5, l6 = [], [], [], [], [], [], [], []

    coltypes = {'Activity_ID': object, 'Quality': object, 'source': object, 'CID': object,
                'SMILES': object, 'connectivity': object, 'InChIKey': object, 'InChI': object,
                'InChI_AuxInfo': object, 'target_id': object, 'accession': object, 'Protein_Type': object,
                'AID': object, 'type_IC50': object, 'type_EC50': object, 'type_KD': object,
                'type_Ki': object, 'type_other': object, 'Activity_class': object, 'relation': object,
                'pchembl_value': object, 'pchembl_value_Mean': np.float64, 'pchembl_value_StdDev': np.float64,
                'pchembl_value_SEM': np.float64, 'pchembl_value_N': np.float64, 'pchembl_value_Median': np.float64,
                'pchembl_value_MAD': np.float64}
    desc_types = {"D001": "int", "D002": "int", "D003": "int", "D004": "int", "D005": "int", "D006": "int",
                  "D007": "int", "D008": "int", "D009": "int", "D010": "int", "D011": "int", "D012": "int",
                  "D013": "int", "D014": "int", "D015": "float", "D016": "int", "D017": "int", "D018": "float",
                  "D019": "int", "D020": "int", "D021": "int", "D022": "int", "D023": "int", "D024": "int",
                  "D025": "int", "D026": "int", "D027": "int", "D028": "int", "D029": "int", "D030": "int",
                  "D031": "int", "D032": "int", "D033": "int", "D034": "int", "D035": "int", "D036": "int",
                  "D037": "int", "D038": "int", "D039": "int", "D040": "int", "D041": "int", "D042": "int",
                  "D043": "int", "D044": "int", "D045": "int", "D046": "int", "D047": "int", "D048": "int",
                  "D049": "int", "D050": "int", "D051": "int", "D052": "int", "D053": "int", "D054": "int",
                  "D055": "int", "D056": "int", "D057": "int", "D058": "int", "D059": "int", "D060": "int",
                  "D061": "int", "D062": "int", "D063": "int", "D064": "int", "D065": "int", "D066": "int",
                  "D067": "int", "D068": "int", "D069": "int", "D070": "int", "D071": "int", "D072": "int",
                  "D073": "int", "D074": "int", "D075": "int", "D076": "int", "D077": "int", "D078": "int",
                  "D079": "int", "D080": "int", "D081": "int", "D082": "int", "D083": "int", "D084": "int",
                  "D085": "int", "D086": "int", "D087": "int", "D088": "int", "D089": "int", "D090": "int",
                  "D091": "int", "D092": "int", "D093": "int", "D094": "int", "D095": "int", "D096": "int",
                  "D097": "int", "D098": "int", "D099": "int", "D100": "int", "D101": "int", "D102": "int",
                  "D103": "int", "D104": "int", "D105": "int", "D106": "int", "D107": "int", "D108": "int",
                  "D109": "int", "D110": "int", "D111": "int", "D112": "int", "D113": "int", "D114": "int",
                  "D115": "int", "D116": "int", "D117": "int", "D118": "int", "D119": "int", "D120": "int",
                  "D121": "int", "D122": "float", "D123": "float", "D124": "int", "D125": "int", "D126": "int",
                  "D127": "int", "D128": "int", "D129": "int", "D130": "int", "D131": "float", "D132": "float",
                  "D133": "float", "D134": "int", "D135": "float", "D136": "float", "D137": "float", "D138": "float",
                  "D139": "int", "D140": "float", "D141": "float", "D142": "float", "D143": "float", "D144": "float",
                  "D145": "float", "D146": "float", "D147": "float", "D148": "float", "D149": "float", "D150": "float",
                  "D151": "float", "D152": "float", "D153": "int", "D154": "float", "D155": "int", "D156": "int",
                  "D157": "float", "D158": "float", "D159": "int", "D160": "float", "D161": "int", "D162": "float",
                  "D163": "float", "D164": "float", "D165": "int", "D166": "float", "D167": "float", "D168": "float",
                  "D169": "float", "D170": "float", "D171": "float", "D172": "float", "D173": "float", "D174": "int",
                  "D175": "int", "D176": "float", "D177": "float", "D178": "int", "D179": "float", "D180": "float",
                  "D181": "float", "D182": "float", "D183": "float", "D184": "float", "D185": "float", "D186": "float",
                  "D187": "float", "D188": "float", "D189": "float", "D190": "float", "D191": "float", "D192": "float",
                  "D193": "float", "D194": "float", "D195": "float", "D196": "float", "D197": "float", "D198": "float",
                  "D199": "float", "D200": "float", "D201": "float", "D202": "float", "D203": "float", "D204": "float",
                  "D205": "float", "D206": "float", "D207": "float", "D208": "float", "D209": "float", "D210": "float",
                  "D211": "float", "D212": "float", "D213": "float", "D214": "float", "D215": "float", "D216": "float",
                  "D217": "float", "D218": "float", "D219": "float", "D220": "float", "D221": "float", "D222": "float",
                  "D223": "float", "D224": "float", "D225": "float", "D226": "float", "D227": "float", "D228": "float",
                  "D229": "float", "D230": "float", "D231": "float", "D232": "float", "D233": "float", "D234": "float",
                  "D235": "float", "D236": "float", "D237": "float", "D238": "float", "D239": "int", "D240": "int",
                  "D241": "float", "D242": "float", "D243": "float", "D244": "int", "D245": "int", "D246": "int",
                  "D247": "float", "D248": "float", "D249": "float", "D250": "float", "D251": "float", "D252": "int",
                  "D253": "float", "D254": "float", "D255": "float", "D256": "float", "D257": "float", "D258": "float",
                  "D259": "float", "D260": "float", "D261": "float", "D262": "float", "D263": "float", "D264": "float",
                  "D265": "float", "D266": "float", "D267": "float", "D268": "float", "D269": "float", "D270": "float",
                  "D271": "float", "D272": "float", "D273": "float", "D274": "float", "D275": "float", "D276": "float",
                  "D277": "float", "D278": "float", "D279": "float", "D280": "float", "D281": "float", "D282": "float",
                  "D283": "float", "D284": "float", "D285": "float", "D286": "float", "D287": "float", "D288": "float",
                  "D289": "float", "D290": "float", "D291": "float", "D292": "float", "D293": "float", "D294": "float",
                  "D295": "float", "D296": "float", "D297": "float", "D298": "float", "D299": "float", "D300": "float",
                  "D301": "float", "D302": "float", "D303": "float", "D304": "float", "D305": "float", "D306": "float",
                  "D307": "float", "D308": "float", "D309": "float", "D310": "float", "D311": "float", "D312": "float",
                  "D313": "float", "D314": "float", "D315": "float", "D316": "float", "D317": "float", "D318": "float",
                  "D319": "float", "D320": "float", "D321": "float", "D322": "float", "D323": "float", "D324": "float",
                  "D325": "float", "D326": "float", "D327": "float", "D328": "float", "D329": "float", "D330": "float",
                  "D331": "float", "D332": "float", "D333": "float", "D334": "float", "D335": "float", "D336": "float",
                  "D337": "float", "D338": "float", "D339": "float", "D340": "float", "D341": "float", "D342": "int",
                  "D343": "int", "D344": "int", "D345": "int", "D346": "int", "D347": "int", "D348": "int",
                  "D349": "int", "D350": "int", "D351": "float", "D352": "float", "D353": "float", "D354": "float",
                  "D355": "float", "D356": "float", "D357": "float", "D358": "float", "D359": "int", "D360": "float",
                  "D361": "float", "D362": "float", "D363": "float", "D364": "float", "D365": "float", "D366": "int",
                  "D367": "int", "D368": "int", "D369": "int", "D370": "int", "D371": "int", "D372": "int",
                  "D373": "int", "D374": "int", "D375": "int", "D376": "int", "D377": "int", "D378": "int",
                  "D379": "int", "D380": "int", "D381": "int", "D382": "int", "D383": "int", "D384": "int",
                  "D385": "int", "D386": "int", "D387": "int", "D388": "int", "D389": "int", "D390": "int",
                  "D391": "int", "D392": "int", "D393": "int", "D394": "int", "D395": "int", "D396": "int",
                  "D397": "int", "D398": "int", "D399": "int", "D400": "int", "D401": "int", "D402": "int",
                  "D403": "int", "D404": "int", "D405": "int", "D406": "int", "D407": "int", "D408": "int",
                  "D409": "int", "D410": "int", "D411": "int", "D412": "int", "D413": "int", "D414": "int",
                  "D415": "float", "D416": "float", "D417": "float", "D418": "float", "D419": "float", "D420": "float",
                  "D421": "float", "D422": "float", "D423": "float", "D424": "float", "D425": "float", "D426": "float",
                  "D427": "float", "D428": "float", "D429": "float", "D430": "float", "D431": "float", "D432": "float",
                  "D433": "float", "D434": "float", "D435": "float", "D436": "float", "D437": "float", "D438": "float",
                  "D439": "float", "D440": "float", "D441": "float", "D442": "float", "D443": "float", "D444": "float",
                  "D445": "float", "D446": "float", "D447": "float", "D448": "float", "D449": "float", "D450": "float",
                  "D451": "float", "D452": "float", "D453": "float", "D454": "float", "D455": "float", "D456": "float",
                  "D457": "float", "D458": "float", "D459": "float", "D460": "float", "D461": "float", "D462": "float",
                  "D463": "float", "D464": "float", "D465": "float", "D466": "float", "D467": "float", "D468": "float",
                  "D469": "float", "D470": "float", "D471": "float", "D472": "float", "D473": "float", "D474": "float",
                  "D475": "float", "D476": "float", "D477": "float", "D478": "float", "D479": "float", "D480": "float",
                  "D481": "float", "D482": "float", "D483": "float", "D484": "float", "D485": "float", "D486": "float",
                  "D487": "float", "D488": "float", "D489": "float", "D490": "float", "D491": "float", "D492": "float",
                  "D493": "float", "D494": "float", "D495": "float", "D496": "float", "D497": "float", "D498": "float",
                  "D499": "float", "D500": "float", "D501": "float", "D502": "float", "D503": "float", "D504": "float",
                  "D505": "float", "D506": "float", "D507": "float", "D508": "float", "D509": "float", "D510": "float",
                  "D511": "float", "D512": "float", "D513": "float", "D514": "float", "D515": "float", "D516": "float",
                  "D517": "float", "D518": "float", "D519": "float", "D520": "float", "D521": "float", "D522": "float",
                  "D523": "float", "D524": "float", "D525": "float", "D526": "float", "D527": "float", "D528": "float",
                  "D529": "float", "D530": "float", "D531": "float", "D532": "float", "D533": "float", "D534": "float",
                  "D535": "float", "D536": "float", "D537": "float", "D538": "float", "D539": "float", "D540": "float",
                  "D541": "float", "D542": "float", "D543": "float", "D544": "float", "D545": "float", "D546": "float",
                  "D547": "float", "D548": "float", "D549": "float", "D550": "float", "D551": "float", "D552": "float",
                  "D553": "float", "D554": "float", "D555": "float", "D556": "float", "D557": "float", "D558": "float",
                  "D559": "float", "D560": "float", "D561": "float", "D562": "float", "D563": "float", "D564": "float",
                  "D565": "float", "D566": "float", "D567": "float", "D568": "float", "D569": "float", "D570": "float",
                  "D571": "float", "D572": "float", "D573": "float", "D574": "float", "D575": "float", "D576": "float",
                  "D577": "float", "D578": "float", "D579": "float", "D580": "float", "D581": "float", "D582": "float",
                  "D583": "float", "D584": "float", "D585": "float", "D586": "float", "D587": "float", "D588": "float",
                  "D589": "float", "D590": "float", "D591": "float", "D592": "float", "D593": "float", "D594": "float",
                  "D595": "float", "D596": "int", "D597": "int", "D598": "int", "D599": "int", "D600": "int",
                  "D601": "int", "D602": "int", "D603": "int", "D604": "int", "D605": "int", "D606": "int",
                  "D607": "int", "D608": "int", "D609": "int", "D610": "int", "D611": "int", "D612": "int",
                  "D613": "int", "D614": "int", "D615": "int", "D616": "int", "D617": "int", "D618": "int",
                  "D619": "int", "D620": "int", "D621": "int", "D622": "int", "D623": "int", "D624": "int",
                  "D625": "int", "D626": "int", "D627": "int", "D628": "int", "D629": "int", "D630": "int",
                  "D631": "int", "D632": "int", "D633": "int", "D634": "int", "D635": "int", "D636": "int",
                  "D637": "int", "D638": "int", "D639": "int", "D640": "int", "D641": "int", "D642": "int",
                  "D643": "int", "D644": "int", "D645": "int", "D646": "int", "D647": "int", "D648": "int",
                  "D649": "int", "D650": "int", "D651": "int", "D652": "int", "D653": "int", "D654": "int",
                  "D655": "int", "D656": "int", "D657": "int", "D658": "int", "D659": "int", "D660": "int",
                  "D661": "int", "D662": "int", "D663": "int", "D664": "int", "D665": "int", "D666": "int",
                  "D667": "int", "D668": "int", "D669": "int", "D670": "int", "D671": "int", "D672": "int",
                  "D673": "int", "D674": "int", "D675": "int", "D676": "int", "D677": "int", "D678": "int",
                  "D679": "int", "D680": "int", "D681": "int", "D682": "int", "D683": "int", "D684": "int",
                  "D685": "int", "D686": "int", "D687": "int", "D688": "int", "D689": "int", "D690": "int",
                  "D691": "int", "D692": "int", "D693": "int", "D694": "int", "D695": "int", "D696": "int",
                  "D697": "int", "D698": "int", "D699": "int", "D700": "int", "D701": "int", "D702": "int",
                  "D703": "int", "D704": "int", "D705": "int", "D706": "int", "D707": "int", "D708": "int",
                  "D709": "int", "D710": "int", "D711": "int", "D712": "int", "D713": "int", "D714": "int",
                  "D715": "int", "D716": "int", "D717": "int", "D718": "int", "D719": "int", "D720": "int",
                  "D721": "int", "D722": "int", "D723": "int", "D724": "int", "D725": "int", "D726": "int",
                  "D727": "int", "D728": "int", "D729": "int", "D730": "int", "D731": "int", "D732": "int",
                  "D733": "int", "D734": "int", "D735": "int", "D736": "int", "D737": "int", "D738": "int",
                  "D739": "int", "D740": "int", "D741": "int", "D742": "int", "D743": "int", "D744": "int",
                  "D745": "int", "D746": "int", "D747": "int", "D748": "int", "D749": "int", "D750": "int",
                  "D751": "int", "D752": "int", "D753": "int", "D754": "int", "D755": "int", "D756": "int",
                  "D757": "int", "D758": "int", "D759": "int", "D760": "int", "D761": "int", "D762": "int",
                  "D763": "int", "D764": "int", "D765": "int", "D766": "int", "D767": "int", "D768": "int",
                  "D769": "int", "D770": "int", "D771": "int", "D772": "int", "D773": "int", "D774": "float",
                  "D775": "float", "D776": "float", "D777": "float"}

    folder = '/zfsdata/data/datasets/PCM_set/creation/version_05.4/release/'
    descriptors = folder + 'descriptors/'

    prot_data = pd.read_csv(folder + '05.4_combined_set_protein_targets.tsv.gz',
                            sep='\t', keep_default_na=False).rename(columns={'TARGET_NAME': 'target_id'})

    prot_descs = pd.read_csv(descriptors + '05.4_combined_prot_embeddings_unirep.tsv.gz',
                             sep='\t', keep_default_na=False).rename(columns={'TARGET_NAME': 'target_id'})

    mol_descs = pd.concat([chunk for chunk in tqdm(
        pd.read_csv(descriptors + '05.4_combined_2D_moldescs_mold2.tsv.gz', chunksize=50000,
                    sep='\t', keep_default_na=False, dtype=desc_types, low_memory=True),
        total=26)], axis=0)
    mol_descs.iloc[:, 1:] = (mol_descs.iloc[:, 1:] * 1000).astype(np.int32)
    mol_descs.iloc[:, 1:] = np.nan_to_num(mol_descs.iloc[:, 1:], copy=False, nan=0, posinf=0, neginf=0)
    mol_descs.iloc[:, 1:][mol_descs.iloc[:, 1:] < 0] = 0
    print(mol_descs.iloc[:, 1:].min().min(), mol_descs.iloc[:, 1:].max().max())
    mol_descs.to_csv("./save_positive_moldescs.txt.xz", index=False, sep='\t')

    for chunk in tqdm(pd.read_csv(folder + '05.4_combined_set_without_stereochemistry.tsv.xz',
                                  sep='\t', chunksize=50000, dtype=coltypes,
                                  usecols=['connectivity', 'target_id', 'Quality', 'Activity_class', 'relation',
                                           'pchembl_value_Mean',
                                           'pchembl_value_StdDev']), total=1196, ncols=80):
        # Obtain PCM descriptors
        chunk = chunk.merge(mol_descs, on='connectivity')
        chunk = chunk.merge(prot_descs, on='target_id')
        fps = [tm.VectorUint(chunk.iloc[i, 7:].tolist()) for i in range(chunk.shape[0])]
        # Add to LSH forest
        lf.batch_add(fps)
        del fps
        # Obtain mol properties for legend
        mols = chunk['SMILES'].apply(lambda x: Chem.MolFromSmiles(x)).tolist()
        descs = get_mol_props_for_map(mols)
        hac.extend(descs[0])
        c_frac.extend(descs[1])
        ring_atom_frac.extend(descs[2])
        ring_size.extend(descs[3])
        # Obtain protein properties for legend
        merged = chunk['target_id'].merge(prot_data, on='target_id')
        # Organisms with > 20k activity values
        allowed_organisms = ["Homo sapiens (Human)", "Mus musculus (Mouse)", "Rattus norvegicus (Rat)",
                             "Escherichia coli (strain K12)", "Equus caballus (Horse)",
                             "Influenza A virus (A/WSN/1933(H1N1))", "Trypanosoma cruzi",
                             "Schistosoma mansoni (Blood fluke)", "Bacillus subtilis"]
        organisms.extend([organism if organism in allowed_organisms else 'Other' for organism in merged['Organism']])
        lengths.extend(merged['Length'])
        # Protein classification
        classif = prot_data['Classification'].str.split(';').apply(lambda x: x[0]).str.split('->')
        l1.extend(classif.apply(lambda x: x[0]))
        l2.extend(classif.apply(lambda x: x[1] if len(x) > 1 else ''))
        l3.extend(classif.apply(lambda x: x[2] if len(x) > 2 else ''))
        l4.extend(classif.apply(lambda x: x[3] if len(x) > 3 else ''))
        l5.extend(classif.apply(lambda x: x[4] if len(x) > 4 else ''))
        l6.extend(classif.apply(lambda x: x[5] if len(x) > 5 else ''))
        del classif
        break

    lf.index()
    lf.store("TEST_Papyrus_lshforest_without_stereo.dat")
    cfg = tm.LayoutConfiguration()
    cfg.node_size = 1 / 260
    cfg.mmm_repeats = 2
    cfg.sl_extra_scaling_steps = 5
    cfg.k = 20
    cfg.sl_scaling_type = tm.RelativeToAvgLength

    organisms_labels, organisms_data = ordered_top_labels(organisms, 9)
    l1_labels, l1_data = ordered_top_labels(l1, 9)
    l2_labels, l2_data = ordered_top_labels(l2, 9)
    l3_labels, l3_data = ordered_top_labels(l3, 9)
    l4_labels, l4_data = ordered_top_labels(l4, 9)
    l5_labels, l5_data = ordered_top_labels(l5, 9)
    l6_labels, l6_data = ordered_top_labels(l6, 9)

    x, y, s, t, _ = tm.layout_from_lsh_forest(lf, cfg)
    f = Faerun(view="front", coords=False, clear_color='#ffffff', alpha_blending=True)
    f.add_scatter("Papyrus_nostereo", {'x': x,
                                       'y': y,
                                       'c': [
                                           hac,
                                           c_frac,
                                           ring_atom_frac,
                                           ring_size,
                                           organisms_data,
                                           lengths,
                                           l1_data,
                                           l2_data,
                                           l3_data,
                                           l4_data,
                                           l5_data,
                                           l6_data
                                       ]}, shader="smoothCircle", point_scale=2.0,
                  max_point_size=20,
                  legend_labels=[organisms_labels,
                                 l1_labels,
                                 l2_labels,
                                 l3_labels,
                                 l4_labels,
                                 l5_labels,
                                 l6_labels],
                  categorical=[False, False, False, False, True, False, True, True, True, True, True, True],
                  colormap=["rainbow", "rainbow", "rainbow", "rainbow", "tab10", "rainbow", "tab10", "tab10", "tab10",
                            "tab10", "tab10", "tab10"],
                  series_title=[
                      "Heavy atom count",
                      "Carbon fraction",
                      "Ring atom fraction",
                      "Largest ring size",
                      "Organism",
                      "Sequence length",
                      "Protein class level 1",
                      "Protein class level 2",
                      "Protein class level 3",
                      "Protein class level 4",
                      "Protein class level 5",
                      "Protein class level 6"],
                  has_legend=True,
                  )
    f.add_tree('papyrus_nostereo', {"from": s, "to": t}, point_helper="Papyrus_nostereo", color="#222222")
    f.plot("TEST_Papyrus_no_stereo")
