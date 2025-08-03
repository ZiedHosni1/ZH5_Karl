import gzip
import math
import pickle
from collections import defaultdict
from copy import deepcopy
from math import exp, log
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import Crippen, Descriptors, PandasTools
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect

rdBase.DisableLog("rdApp.error")

# ============================================================================
# Build the vocabulary for SMILES. Besides, definite vectorize function (atoms -> numerics) and devectorize function (numerics -> atoms)


class Tokenizer:
    def __init__(self):
        self.start = "^"
        self.end = "$"
        self.pad = " "

    def build_vocab(self):
        chars = []
        # atoms (carbon), replace Cl for Q and Br for W
        chars = chars + [
            "H",
            "B",
            "c",
            "C",
            "n",
            "N",
            "o",
            "O",
            "p",
            "P",
            "s",
            "S",
            "F",
            "Q",
            "W",
            "I",
        ]
        # hydrogens: H2 to Z, H3 to X
        chars = chars + ["[", "]", "+", "Z", "X"]
        # bounding
        chars = chars + ["-", "=", "#", "."]
        # branches
        chars = chars + ["(", ")"]
        # cycles
        chars = chars + ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]
        # anit/clockwise
        chars = chars + ["@"]
        # directional bonds
        chars = chars + ["/", "\\"]
        # 10+ rings
        chars = chars + ["%"]
        # Important that pad gets value 0
        self.tokenlist = [self.pad, self.start, self.end] + list(chars)

    @property
    def tokenlist(self):
        return self._tokenlist

    @tokenlist.setter
    def tokenlist(self, tokenlist):
        self._tokenlist = tokenlist
        # create the dictionaries
        self.char_to_int = {c: i for i, c in enumerate(self._tokenlist)}
        self.int_to_char = {i: c for c, i in self.char_to_int.items()}

    def encode(self, smi):
        encoded = []
        smi = smi.replace("Cl", "Q")
        smi = smi.replace("Br", "W")
        # hydrogens
        smi = smi.replace("H2", "Z")
        smi = smi.replace("H3", "X")

        return (
            [self.char_to_int[self.start]]
            + [self.char_to_int[s] for s in smi]
            + [self.char_to_int[self.end]]
        )

    def decode(self, ords):
        smi = "".join([self.int_to_char[o] for o in ords])
        # hydrogens
        smi = smi.replace("Z", "H2")
        smi = smi.replace("X", "H3")
        # replace proxy atoms for double char atoms symbols
        smi = smi.replace("Q", "Cl")
        smi = smi.replace("W", "Br")

        return smi

    @property
    def n_tokens(self):
        return len(self.int_to_char)


# ============================================================================
# Select chemical properties
def reward_fn(properties, generated_smiles):
    if properties == "druglikeness":
        vals = batch_druglikeness(generated_smiles)
    elif properties == "solubility":
        vals = batch_solubility(generated_smiles)
    elif properties == "synthesizability":
        vals = batch_SA(generated_smiles)
    elif properties == "nhoc":
        vals = batch_nhoc(generated_smiles)
    elif properties == "vol_nhoc":
        vals = batch_vol_nhoc(generated_smiles)
    return vals


# Diversity
def batch_diversity(smiles):
    scores = []
    df = pd.DataFrame({"smiles": smiles})
    PandasTools.AddMoleculeColumnToFrame(df, "smiles", "mol")
    fps = [
        GetMorganFingerprintAsBitVect(m, 4, nBits=2048)
        for m in df["mol"]
        if m is not None
    ]
    for i in range(1, len(fps)):
        scores.extend(
            DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i], returnDistance=True)
        )
    return np.mean(scores)


# ============================================================================
# Druglikeness
AliphaticRings = Chem.MolFromSmarts("[$([A;R][!a])]")
AcceptorSmarts = [
    "[oH0;X2]",
    "[OH1;X2;v2]",
    "[OH0;X2;v2]",
    "[OH0;X1;v2]",
    "[O-;X1]",
    "[SH0;X2;v2]",
    "[SH0;X1;v2]",
    "[S-;X1]",
    "[nH0;X2]",
    "[NH0;X1;v3]",
    "[$([N;+0;X3;v3]);!$(N[C,S]=O)]",
]
Acceptors = []
for hba in AcceptorSmarts:
    Acceptors.append(Chem.MolFromSmarts(hba))
StructuralAlertSmarts = [
    "*1[O,S,N]*1",
    "[S,C](=[O,S])[F,Br,Cl,I]",
    "[CX4][Cl,Br,I]",
    "[C,c]S(=O)(=O)O[C,c]",
    "[$([CH]),$(CC)]#CC(=O)[C,c]",
    "[$([CH]),$(CC)]#CC(=O)O[C,c]",
    "n[OH]",
    "[$([CH]),$(CC)]#CS(=O)(=O)[C,c]",
    "C=C(C=O)C=O",
    "n1c([F,Cl,Br,I])cccc1",
    "[CH1](=O)",
    "[O,o][O,o]",
    "[C;!R]=[N;!R]",
    "[N!R]=[N!R]",
    "[#6](=O)[#6](=O)",
    "[S,s][S,s]",
    "[N,n][NH2]",
    "C(=O)N[NH2]",
    "[C,c]=S",
    "[$([CH2]),$([CH][CX4]),$(C([CX4])[CX4])]=[$([CH2]),$([CH][CX4]),$(C([CX4])[CX4])]",
    "C1(=[O,N])C=CC(=[O,N])C=C1",
    "C1(=[O,N])C(=[O,N])C=CC=C1",
    "a21aa3a(aa1aaaa2)aaaa3",
    "a31a(a2a(aa1)aaaa2)aaaa3",
    "a1aa2a3a(a1)A=AA=A3=AA=A2",
    "c1cc([NH2])ccc1",
    "[Hg,Fe,As,Sb,Zn,Se,se,Te,B,Si,Na,Ca,Ge,Ag,Mg,K,Ba,Sr,Be,Ti,Mo,Mn,Ru,Pd,Ni,Cu,Au,Cd,Al,Ga,Sn,Rh,Tl,Bi,Nb,Li,Pb,Hf,Ho]",
    "I",
    "OS(=O)(=O)[O-]",
    "[N+](=O)[O-]",
    "C(=O)N[OH]",
    "C1NC(=O)NC(=O)1",
    "[SH]",
    "[S-]",
    "c1ccc([Cl,Br,I,F])c([Cl,Br,I,F])c1[Cl,Br,I,F]",
    "c1cc([Cl,Br,I,F])cc([Cl,Br,I,F])c1[Cl,Br,I,F]",
    "[CR1]1[CR1][CR1][CR1][CR1][CR1][CR1]1",
    "[CR1]1[CR1][CR1]cc[CR1][CR1]1",
    "[CR2]1[CR2][CR2][CR2][CR2][CR2][CR2][CR2]1",
    "[CR2]1[CR2][CR2]cc[CR2][CR2][CR2]1",
    "[CH2R2]1N[CH2R2][CH2R2][CH2R2][CH2R2][CH2R2]1",
    "[CH2R2]1N[CH2R2][CH2R2][CH2R2][CH2R2][CH2R2][CH2R2]1",
    "C#C",
    "[OR2,NR2]@[CR2]@[CR2]@[OR2,NR2]@[CR2]@[CR2]@[OR2,NR2]",
    "[$([N+R]),$([n+R]),$([N+]=C)][O-]",
    "[C,c]=N[OH]",
    "[C,c]=NOC=O",
    "[C,c](=O)[CX4,CR0X3,O][C,c](=O)",
    "c1ccc2c(c1)ccc(=O)o2",
    "[O+,o+,S+,s+]",
    "N=C=O",
    "[NX3,NX4][F,Cl,Br,I]",
    "c1ccccc1OC(=O)[#6]",
    "[CR0]=[CR0][CR0]=[CR0]",
    "[C+,c+,C-,c-]",
    "N=[N+]=[N-]",
    "C12C(NC(N1)=O)CSC2",
    "c1c([OH])c([OH,NH2,NH])ccc1",
    "P",
    "[N,O,S]C#N",
    "C=C=O",
    "[Si][F,Cl,Br,I]",
    "[SX2]O",
    "[SiR0,CR0](c1ccccc1)(c2ccccc2)(c3ccccc3)",
    "O1CCCCC1OC2CCC3CCCCC3C2",
    "N=[CR0][N,n,O,S]",
    "[cR2]1[cR2][cR2]([Nv3X3,Nv4X4])[cR2][cR2][cR2]1[cR2]2[cR2][cR2][cR2]([Nv3X3,Nv4X4])[cR2][cR2]2",
    "C=[C!r]C#N",
    "[cR2]1[cR2]c([N+0X3R0,nX3R0])c([N+0X3R0,nX3R0])[cR2][cR2]1",
    "[cR2]1[cR2]c([N+0X3R0,nX3R0])[cR2]c([N+0X3R0,nX3R0])[cR2]1",
    "[cR2]1[cR2]c([N+0X3R0,nX3R0])[cR2][cR2]c1([N+0X3R0,nX3R0])",
    "[OH]c1ccc([OH,NH2,NH])cc1",
    "c1ccccc1OC(=O)O",
    "[SX2H0][N]",
    "c12ccccc1(SC(S)=N2)",
    "c12ccccc1(SC(=S)N2)",
    "c1nnnn1C=O",
    "s1c(S)nnc1NC=O",
    "S1C=CSC1=S",
    "C(=O)Onnn",
    "OS(=O)(=O)C(F)(F)F",
    "N#CC[OH]",
    "N#CC(=O)",
    "S(=O)(=O)C#N",
    "N[CH2]C#N",
    "C1(=O)NCC1",
    "S(=O)(=O)[O-,OH]",
    "NC[F,Cl,Br,I]",
    "C=[C!r]O",
    "[NX2+0]=[O+0]",
    "[OR0,NR0][OR0,NR0]",
    "C(=O)O[C,H1].C(=O)O[C,H1].C(=O)O[C,H1]",
    "[CX2R0][NX3R0]",
    "c1ccccc1[C;!R]=[C;!R]c2ccccc2",
    "[NX3R0,NX4R0,OR0,SX2R0][CX4][NX3R0,NX4R0,OR0,SX2R0]",
    "[s,S,c,C,n,N,o,O]~[n+,N+](~[s,S,c,C,n,N,o,O])(~[s,S,c,C,n,N,o,O])~[s,S,c,C,n,N,o,O]",
    "[s,S,c,C,n,N,o,O]~[nX3+,NX3+](~[s,S,c,C,n,N])~[s,S,c,C,n,N]",
    "[*]=[N+]=[*]",
    "[SX3](=O)[O-,OH]",
    "N#N",
    "F.F.F.F",
    "[R0;D2][R0;D2][R0;D2][R0;D2]",
    "[cR,CR]~C(=O)NC(=O)~[cR,CR]",
    "C=!@CC=[O,S]",
    "[#6,#8,#16][C,c](=O)O[C,c]",
    "c[C;R0](=[O,S])[C,c]",
    "c[SX2][C;!R]",
    "C=C=C",
    "c1nc([F,Cl,Br,I,S])ncc1",
    "c1ncnc([F,Cl,Br,I,S])c1",
    "c1nc(c2c(n1)nc(n2)[F,Cl,Br,I])",
    "[C,c]S(=O)(=O)c1ccc(cc1)F",
    "[15N]",
    "[13C]",
    "[18O]",
    "[34S]",
]
StructuralAlerts = []
for smarts in StructuralAlertSmarts:
    StructuralAlerts.append(Chem.MolFromSmarts(smarts))
pads1 = [
    [
        2.817065973,
        392.5754953,
        290.7489764,
        2.419764353,
        49.22325677,
        65.37051707,
        104.9805561,
    ],
    [
        0.486849448,
        186.2293718,
        2.066177165,
        3.902720615,
        1.027025453,
        0.913012565,
        145.4314800,
    ],
    [
        2.948620388,
        160.4605972,
        3.615294657,
        4.435986202,
        0.290141953,
        1.300669958,
        148.7763046,
    ],
    [
        1.618662227,
        1010.051101,
        0.985094388,
        0.000000001,
        0.713820843,
        0.920922555,
        258.1632616,
    ],
    [
        1.876861559,
        125.2232657,
        62.90773554,
        87.83366614,
        12.01999824,
        28.51324732,
        104.5686167,
    ],
    [
        0.010000000,
        272.4121427,
        2.558379970,
        1.565547684,
        1.271567166,
        2.758063707,
        105.4420403,
    ],
    [
        3.217788970,
        957.7374108,
        2.274627939,
        0.000000001,
        1.317690384,
        0.375760881,
        312.3372610,
    ],
    [
        0.010000000,
        1199.094025,
        -0.09002883,
        0.000000001,
        0.185904477,
        0.875193782,
        417.7253140,
    ],
]
pads2 = [
    [
        2.817065973,
        392.5754953,
        290.7489764,
        2.419764353,
        49.22325677,
        65.37051707,
        104.9805561,
    ],
    [
        3.172690585,
        137.8624751,
        2.534937431,
        4.581497897,
        0.822739154,
        0.576295591,
        131.3186604,
    ],
    [
        2.948620388,
        160.4605972,
        3.615294657,
        4.435986202,
        0.290141953,
        1.300669958,
        148.7763046,
    ],
    [
        1.618662227,
        1010.051101,
        0.985094388,
        0.000000001,
        0.713820843,
        0.920922555,
        258.1632616,
    ],
    [
        1.876861559,
        125.2232657,
        62.90773554,
        87.83366614,
        12.01999824,
        28.51324732,
        104.5686167,
    ],
    [
        0.010000000,
        272.4121427,
        2.558379970,
        1.565547684,
        1.271567166,
        2.758063707,
        105.4420403,
    ],
    [
        3.217788970,
        957.7374108,
        2.274627939,
        0.000000001,
        1.317690384,
        0.375760881,
        312.3372610,
    ],
    [
        0.010000000,
        1199.094025,
        -0.09002883,
        0.000000001,
        0.185904477,
        0.875193782,
        417.7253140,
    ],
]


def ads(x, a, b, c, d, e, f, dmax):
    return (
        a
        + (
            b
            / (1 + exp(-1 * (x - c + d / 2) / e))
            * (1 - 1 / (1 + exp(-1 * (x - c - d / 2) / f)))
        )
    ) / dmax


def qed_eval(w, p, gerebtzoff):
    d = [0.00] * 8
    if gerebtzoff:
        for i in range(0, 8):
            d[i] = ads(
                p[i],
                pads1[i][0],
                pads1[i][1],
                pads1[i][2],
                pads1[i][3],
                pads1[i][4],
                pads1[i][5],
                pads1[i][6],
            )
    else:
        for i in range(0, 8):
            d[i] = ads(
                p[i],
                pads2[i][0],
                pads2[i][1],
                pads2[i][2],
                pads2[i][3],
                pads2[i][4],
                pads2[i][5],
                pads2[i][6],
            )
    t = 0.0
    for i in range(0, 8):
        t += w[i] * log(d[i])
    return exp(t / sum(w))


def qed(mol):
    matches = []
    if mol is None:
        raise WrongArgument("properties(mol)", "mol argument is 'None'")
    x = [0] * 9
    x[0] = Descriptors.MolWt(mol)
    x[1] = Descriptors.MolLogP(mol)
    for hba in Acceptors:
        if mol.HasSubstructMatch(hba):
            matches = mol.GetSubstructMatches(hba)
            x[2] += len(matches)
    x[3] = Descriptors.NumHDonors(mol)
    x[4] = Descriptors.TPSA(mol)
    x[5] = Descriptors.NumRotatableBonds(mol)
    x[6] = Chem.GetSSSR(Chem.DeleteSubstructs(deepcopy(mol), AliphaticRings))
    for alert in StructuralAlerts:
        if mol.HasSubstructMatch(alert):
            x[7] += 1
    ro5_failed = 0
    if x[3] > 5:
        ro5_failed += 1
    if x[2] > 10:
        ro5_failed += 1
    if x[0] >= 500:
        ro5_failed += 1
    if x[1] > 5:
        ro5_failed += 1
    x[8] = ro5_failed
    return qed_eval([0.66, 0.46, 0.05, 0.61, 0.06, 0.65, 0.48, 0.95], x, True)


def batch_druglikeness(smiles):
    vals = []
    for sm in smiles:
        mol = Chem.MolFromSmiles(sm)
        if mol is None:
            vals.append(0.0)
        else:
            val = qed(mol)
            vals.append(val)
    return vals


# Solubility
def batch_solubility(smiles):
    vals = []
    for sm in smiles:
        mol = Chem.MolFromSmiles(sm)
        if mol is None:
            vals.append(0.0)
        else:
            low_logp = -2.12178879609
            high_logp = 6.0429063424
            logp = Crippen.MolLogP(mol)
            val = (logp - low_logp) / (high_logp - low_logp)
            val = np.clip(val, 0.1, 1.0)
            vals.append(val)
    return vals


# Read synthesizability model
def readSAModel(filename="SA_score.pkl.gz"):
    model_data = pickle.load(gzip.open(filename))
    outDict = {}
    for i in model_data:
        for j in range(1, len(i)):
            outDict[i[j]] = float(i[0])
    SA_model = outDict
    return SA_model


SA_model = readSAModel()


# synthesizability
def batch_SA(smiles):
    vals = []
    for sm in smiles:
        mol = Chem.MolFromSmiles(sm)
        if sm != "" and mol is not None and mol.GetNumAtoms() > 1:
            # fragment score
            fp = Chem.AllChem.GetMorganFingerprint(mol, 2)
            fps = fp.GetNonzeroElements()
            score1 = 0.0
            nf = 0
            for bitId, v in fps.items():
                nf += v
                sfp = bitId
                score1 += SA_model.get(sfp, -4) * v
            score1 /= nf
            # features score
            nAtoms = mol.GetNumAtoms()
            nChiralCenters = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
            ri = mol.GetRingInfo()
            nSpiro = Chem.AllChem.CalcNumSpiroAtoms(mol)
            nBridgeheads = Chem.AllChem.CalcNumBridgeheadAtoms(mol)
            nMacrocycles = 0
            for x in ri.AtomRings():
                if len(x) > 8:
                    nMacrocycles += 1
            sizePenalty = nAtoms**1.005 - nAtoms
            stereoPenalty = math.log10(nChiralCenters + 1)
            spiroPenalty = math.log10(nSpiro + 1)
            bridgePenalty = math.log10(nBridgeheads + 1)
            macrocyclePenalty = 0.0
            if nMacrocycles > 0:
                macrocyclePenalty = math.log10(2)
            score2 = (
                0.0
                - sizePenalty
                - stereoPenalty
                - spiroPenalty
                - bridgePenalty
                - macrocyclePenalty
            )
            score3 = 0.0
            if nAtoms > len(fps):
                score3 = math.log(float(nAtoms) / len(fps)) * 0.5
            sascore = score1 + score2 + score3
            min = -4.0
            max = 2.5
            sascore = 11.0 - (sascore - min + 1) / (max - min) * 9.0
            # smooth the 10-end
            if sascore > 8.0:
                sascore = 8.0 + math.log(sascore + 1.0 - 9.0)
            if sascore > 10.0:
                sascore = 10.0
            elif sascore < 1.0:
                sascore = 1.0
            val = (sascore - 5) / (1.5 - 5)
            val = np.clip(val, 0.1, 1.0)
            vals.append(val)
        else:
            vals.append(0.0)
    return vals


# -------------------------------------------------------------------------
# Marrero-Gani group contribution method for NHOC and vol_nhoc
# -------------------------------------------------------------------------
# 1 Group-contribution tables

MG1: Dict[str, Tuple[float, ...]] = {
    # key              Tm      Tb      Tc       Pc        Vc      Hvap      Hf
    "CH3": (0.6953, 0.8491, 1.7506, 0.018615, 68.35, 0.217, -42.479),
    "CH2": (0.2515, 0.7141, 1.3327, 0.013547, 56.28, 4.910, -20.829),
    "CH": (-0.3730, 0.2925, 0.5960, 0.007259, 37.50, 7.962, -7.122),
    "C": (0.0256, -0.0671, 0.0306, 0.001219, 16.01, 10.730, 8.928),
    # olefinic
    "CH2=CH": (1.1728, 1.5596, 3.2295, 0.025745, 111.43, 4.031, 57.509),
    "CH=CH": (0.9460, 1.5597, 3.0741, 0.023003, 98.43, 9.456, 69.664),
    "CH2=C": (0.7662, 1.3621, 2.7717, 0.021137, 91.40, 8.602, 61.625),
    "CH=C": (0.1732, 1.2971, 2.5666, 0.019609, 83.89, 14.095, 81.835),
    "C=C": (0.3928, 1.2739, 2.6391, 0.014114, 90.66, 19.910, 95.710),
    # acetylenic
    "CH#C": (2.2276, 1.7618, 3.7897, 0.014010, 84.60, 6.144, 224.902),
    "C#C": (2.0516, 1.6767, 4.5870, 0.010888, 74.66, 12.540, 228.282),
    # saturated rings
    "CH2(cyc)": (0.5699, 0.8234, 1.8815, 0.009884, 49.24, 3.341, -18.575),
    "CH(cyc)": (0.0335, 0.5946, 1.1020, 0.007596, 44.95, 6.416, -12.464),
    "C(cyc)": (0.1695, 0.0386, -0.2399, 0.003268, 33.32, 7.017, -2.098),
    # cyclo‑alkenes
    "CH=CH(cyc)": (1.1936, 1.5985, 3.6426, 0.013815, 83.91, 7.767, 59.841),
    "CH=C(cyc)": (0.4344, 1.2529, 3.5475, 0.010576, 70.98, 7.171, 64.295),
    "CH2=C(cyc)": (0.2220, 1.5109, 4.4913, 0.019101, 83.96, 5.351, 0),
    # special ring‑pair (ignored in SMARTS counting – still available via
    "ACH": (0.5860, 0.8365, 2.0337, 0.007260, 42.39, 3.683, 12.861),
    "AC_subst": (0.9176, 1.5468, 4.5344, 0.012859, 26.47, 6.824, 24.701),
    "AC_fused_ar": (1.8955, 1.7324, 5.4979, 0.003564, 35.71, 6.631, 20.187),
    "AC_fused_nonar": (1.2065, 1.1995, 3.1058, 0.006512, 34.65, 6.152, 30.768),
    "CH(bicyc)": (0.6647, 0.1415, 0.4963, -0.000985, -3.33, 0.223, 0),
    "C(bicyc)": (0.0792, 0.2019, 1.6480, -0.010560, -12.10, -2.030, 0),
    "CH(spiro)": (0.7730, 0.2900, 1.3500, -0.006200, -4.50, -2.600, 0),
    "C(spiro)": (0.1020, 0.2300, 1.7900, -0.011100, -13.20, -2.300, 0),
}
MG2: Dict[str, Tuple[float, ...]] = {
    # branched paraffins
    "(CH3)2CH": (0.1175, -0.0035, -0.0471, 0.000473, 1.71, -0.399, -0.419),
    "(CH3)3C": (-0.1214, 0.0072, -0.1778, 0.000340, 3.14, -0.417, -1.967),
    "CH(CH3)CH(CH3)": (0.2390, 0.3160, 0.5602, -0.003207, -3.75, 0.532, 6.065),
    "CH(CH3)C(CH3)2": (-0.3276, 0.3976, 0.8994, -0.008733, -10.06, 0.623, 8.078),
    "C(CH3)2C(CH3)2": (3.3297, 0.4487, 1.5535, -0.016852, -8.70, 5.086, 10.535),
    # diene / alkene adjacency
    "diene_adj": (0.7451, 0.1097, 0.4214, 0.000792, -7.88, 1.632, -11.786),
    "CH3-alkene": (0.0524, 0.0369, -0.0172, -0.000101, 0.50, 0.064, -0.048),
    "CH2-alkene": (-0.1077, -0.0537, 0.0262, 0.000815, 0.14, -0.060, 1.449),
    "CH-alkene": (-0.2485, -0.0093, -0.1526, -0.000163, -2.67, 0.004, 3.964),
    # alicyclic substitutions
    "Ccyc-CH2": (-1.9233, 0.0319, 0.1090, -0.000610, -5.17, 0.585, 21.498),
    "Ccyc-CH3": (0.1737, 0.0722, 0.1607, 0.001235, 1.95, 0.808, 0.238),
    "CHcyc-CH3": (-0.1326, -0.1210, -0.1233, 0.000779, 2.79, 0.096, 4.452),
    "CHcyc-CH2": (-0.4669, -0.0148, 0.3816, 0.001694, -2.95, -0.428, 4.428),
    "CHcyc-CH": (-0.3548, 0.1395, 0.1093, 0.000124, 6.19, 0.153, -4.128),
    "CHcyc-C": (-0.1727, 0.1829, 0, 0, 0, 0, 0),
    "CHcyc-CH=CH": (0.6817, -0.1192, 0.0000, 0.000000, -16.97, 6.768, 10.390),
    "CHcyc-C=CH": (-1.0631, -0.0455, -0.2832, 0.002114, -16.97, 0.000, 10.390),
    "AROMRINGs1s2": (-0.6388, -0.1590, -0.3161, 0.000522, 2.86, 1.164, 1.486),
}

MG3: Dict[str, Tuple[float, ...]] = {
    "CHcyc-Chcyc": (0.5460, 0.4387, 2.1761, 0.002745, 7.72, 0.0, -66.870),
    "CHcyc-(CHn)m-CHcyc": (0.4497, 0.5632, 0.0000, 0.0000, 0.00, 0.000, 0.0000),
    "CH_multiring": (0.6647, 0.1415, 0.4963, -0.000985, -3.33, 0.223, 0.0),
    "C_multiring": (0.0792, 0.0000, 0.0000, 0.000000, 0.00, 0.000, 0.000),
    "AROMFUSED[2]": (0.2825, 0.0441, -1.0095, -0.001332, -6.88, 0.694, 1.904),
    "AROMFUSED[3]": (1.6600, 0.0402, -1.0430, 0.004695, 35.21, 1.176, 5.819),
    "AROMFUSED[4p]": (-1.5856, 0.9126, 2.8885, 0.007280, -24.02, -3.417, -19.089),
    "BICYC>C<": (0.5500, 0.0700, 0.8900, -0.004400, -6.95, 0.000, 0.0000),  # Osmont 06
}

# -------------------------------------------------------------------------
# 2 First‑order atom‑level classifier for MG1


def _mg1_counts(mol: Chem.Mol) -> Dict[str, int]:
    """Return a mapping {group: count} by classifying every carbon atom.

    *   classification logic follows Marrero–Gani’s definition of groups;
    *   only MG1 groups are handled here – MG2/MG3 are left to the legacy
        substring heuristic because they depend on *adjacency* patterns
        not expressible via single‑atom typing.
    """
    counts: Dict[str, int] = defaultdict(int)  # type: ignore[arg-type]

    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "C":
            continue  # ignore heteroatoms

        n_H = atom.GetTotalNumHs()
        in_ring = atom.IsInRing()
        has_double = any(
            b.GetBondType() == Chem.BondType.DOUBLE for b in atom.GetBonds()
        )
        has_triple = any(
            b.GetBondType() == Chem.BondType.TRIPLE for b in atom.GetBonds()
        )

        # aromatic carbons (Marrero-Gani: ACH, AC variants)
        if atom.GetIsAromatic():
            if n_H >= 1:
                counts["ACH"] += 1  # first-order aromatic CH
                continue

            # Substituted aromatic C: need to know whether it is a fusion atom
            ri = mol.GetRingInfo()
            atom_idx = atom.GetIdx()
            # how many aromatic rings contian this atom?
            aro_rings = [
                set(r)
                for r in ri.AtomRings()
                if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r)
            ]
            n_arom_memberships = sum(atom_idx in r for r in aro_rings)

            # whether this atom is also in non-aromatic ring
            nonaro_rings = [
                set(r)
                for r in ri.AtomRings()
                if not all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r)
            ]
            in_nonaro = any(atom_idx in r for r in nonaro_rings)

            if n_arom_memberships >= 2:
                # shared by two aromatic rings (e.g. nafthalene)
                counts["AC_fused_ar"] += 1
            elif in_nonaro:
                # shared by aromatic + non-aromatic ring
                counts["AC_fused_nonar"] += 1
            else:
                counts["AC_subst"] += (
                    1  # substituted aromatic carbon (not a fusion atom)
                )
            continue

        # saturated acyclic --------------------------------------------------
        if not in_ring and not (has_double or has_triple):
            if n_H == 3:
                counts["CH3"] += 1
            elif n_H == 2:
                counts["CH2"] += 1
            elif n_H == 1:
                counts["CH"] += 1
            else:
                counts["C"] += 1
            continue

        # unsaturated acyclic (double bond) ----------------------------------
        if not in_ring and has_double and not has_triple:
            if n_H == 2:
                counts["CH2=CH"] += 1
            elif n_H == 1:
                counts["CH=CH"] += 1
            elif n_H == 0:
                counts["C=C"] += 1
            elif n_H < 0:
                pass
            continue

        # acetylenic ---------------------------------------------------------
        if not in_ring and has_triple:
            if n_H == 1:
                counts["CH#C"] += 1
            else:
                counts["C#C"] += 1
            continue

        # saturated cyclic ---------------------------------------------------
        if in_ring and not (has_double or has_triple):
            if n_H == 2:
                counts["CH2(cyc)"] += 1
            elif n_H == 1:
                counts["CH(cyc)"] += 1
            else:
                counts["C(cyc)"] += 1
            continue

        # unsaturated cyclic (single double bond) ----------------------------
        if in_ring and has_double and not has_triple:
            if n_H == 2:
                counts["CH2=C(cyc)"] += 1
            elif n_H == 1:
                counts["CH=CH(cyc)"] += 1
            else:
                counts["CH=C(cyc)"] += 1
            continue

    # other fall through and are ignored
    return counts


def _mg2_mg3_counts(mol: Chem.Mol) -> Dict[str, int]:
    """Count a small set of high-leverage MG2/MG3 structural motifs."""
    counts = defaultdict(int)
    ri = mol.GetRingInfo()

    # MG2: AROMRINGs1s2 (adjacent substituents on aromatic ring)
    # count rings with two adjacaent sp2 carbons both substituted (not ACH)
    for ring in ri.AtomRings():
        ring_atoms = [mol.GetAtomWithIdx(i) for i in ring]
        if not all(a.GetIsAromatic() for a in ring_atoms):
            continue
        ring_set = set(ring)
        # find substituted aromatic C (no H) inside the ring
        subs = [
            a.GetIdx()
            for a in ring_atoms
            if a.GetIsAromatic() and a.GetTotalNumHs() == 0
        ]
        subs_set = set(subs)
        # adjacency along the ring (edges within the ring)
        for i in range(len(ring)):
            a = ring[i]
            b = ring[(i + 1) % len(ring)]
            if a in subs_set and b in subs_set:
                if "AROMRINGs1s2" in MG2:
                    counts["AROMRINGs1s2"] += 1

    # MG3: CHcyc-Chcyc (fused/alicyclic rings in saturated systems)
    # count ring bonds were both atoms are sp3, in rings, non-aromatic, and each bears one H
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtom(), bond.GetEndAtom()
        if bond.IsInRing() and (not a.GetIsAromatic()) and (not b.GetIsAromatic()):
            if a.IsInRing() and b.IsInRing():
                if (
                    a.GetHybridization().name == "SP3"
                    and b.GetHybridization().name == "SP3"
                ):
                    if a.GetTotalNumHs() == 1 and b.GetTotalNumHs() == 1:
                        if "CHcyc-Chcyc" in MG3:
                            counts["CHcyc-Chcyc"] += 1

    # ---- MG3: fused aromatic systems (AROMFUSED[2],[3],[4p])
    # Build a graph of aromatic rings, edges = shared edge (>= 2 shared atoms)
    aro_rings = [
        set(r)
        for r in ri.AtomRings()
        if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r)
    ]
    seen = set()
    for i in range(len(aro_rings)):
        if i in seen:
            continue
        comp = {i}
        edges = 0
        stack = [i]
        while stack:
            u = stack.pop()
            for v in range(len(aro_rings)):
                if v == u:
                    continue
                if len(aro_rings[u] & aro_rings[v]) >= 2:  # fused by an edge
                    edges += 1
                    if v not in comp:
                        comp.add(v)
                        stack.append(v)
        seen |= comp
        n = len(comp)
        if n == 2 and "AROMFUSED[2]" in MG3:
            counts["AROMFUSED[2]"] += 1
        elif n == 3 and "AROMFUSED[3]" in MG3:
            counts["AROMFUSED[3]"] += 1
        elif n >= 4 and "AROMFUSED[4p]" in MG3:
            if edges >= n:
                counts["AROMFUSED[4p]"] += 1
            elif "AROMFUSED[3]" in MG3 and n == 4:
                counts["AROMFUSED[3]"] += 1  # catacondensed 4-ring

        # MG3: CH_multiring / C_multiring -------------------------------
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "C" or atom.GetIsAromatic():
            continue
        # number of non-aromatic rings that contain this atom
        memberships = sum(
            atom.GetIdx() in r
            for r in ri.AtomRings()
            if not all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r)
        )
        if memberships >= 2:  # shared by >=2 rings
            if atom.GetTotalNumHs() == 1 and "CH_multiring" in MG3:
                counts["CH_multiring"] += 1
            elif atom.GetTotalNumHs() == 0 and "C_multiring" in MG3:
                counts["C_multiring"] += 1

    return counts


# -------------------------------------------------------------------------
# 2b Atom counters - utility needed later bor NHOC


def _count_atoms(mol: Chem.Mol) -> Tuple[int, int]:
    """Return (nC, nH) counting *implicit* and *explicit* hydrogens."""
    nC = nH = 0
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym == "C":
            nC += 1
        elif sym == "H":
            # explicit H (rare in SMILES for hydrocarbons)
            nH += 1
        # implicit H on every heavy atom
        nH += atom.GetTotalNumHs()
    return nC, nH


# -------------------------------------------------------------------------
# 3 Group-contribution engine
# -------------------------------------------------------------------------

_IDX_VC = 4  # tuple indices shared by all MG tables
_IDX_HF = 6

# Heats of formation of reference species (kJ mol-1)
_DH_F_CO2 = -395.51  # CO2(l)
_DH_F_H2O = -241.83  # H2O(l)
_HF0 = 5.549  # universal correction from MG Table 2


def _group_contribution(smiles: str, idx: int) -> float:
    """Return sum(n_i G_i) for the requested property column idx."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0

    # MG1 via RDKit ------------------------------------
    counts1 = _mg1_counts(mol)
    total = 0.0
    for g, n in counts1.items():
        tup = MG1.get(g)  # None if group not in the dict
        if tup is None:
            continue  # silently ignore unknown group
        total += n * tup[idx]

        # MG2/MG3 via structural counter -----------------------------------
    counts_high = _mg2_mg3_counts(mol)
    for g, n in counts_high.items():
        if g in MG2:
            total += n * MG2[g][idx]
        elif g in MG3:
            total += n * MG3[g][idx]

    # MG2/MG3 via low-cost substring fallback --------------------------
    for d in (globals().get("MG2", {}), globals().get("MG3", {})):
        for grp, tup in d.items():
            n = smiles.count(grp)
            if n:
                total += n * tup[idx]

    return total


# universal gas constant in cm3 bar mol-1 K-1 --------------------------
_R_BAR_CM3 = 83.14472

# column indices (consistent with MG1/MG2/MG3 layout)
_IDX_TB = 1
_IDX_TC = 2
_IDX_PC = 3


def _tb(smiles):
    S = _group_contribution(smiles, _IDX_TB)
    if S <= 0:  # avoid log-domain error
        return 0.0
    return 222.543 * math.log(S)


def _tc(smiles):
    S = _group_contribution(smiles, _IDX_TC)
    if S <= 0:  # avoid log-domain error
        return 0.0
    return 231.239 * math.log(S)


# Pc (critical pressure, bar) -----------------------------------------
def _pc(smiles: str) -> float:
    S = _group_contribution(smiles, _IDX_PC)  # sum( n_i G_i)  (bar^{-1/2} units)
    return 5.9827 + 1.0 / (S + 0.108998) ** 2


def _acentric(smiles: str) -> float:
    tb, tc, pc = _tb(smiles), _tc(smiles), _pc(smiles)
    if tc <= 0 or tb <= 0 or pc <= 0 or tb >= tc:
        return 0.0
    ratio = tb / tc
    return (3.0 / 7.0) * ratio / (1.0 - ratio) * math.log10(pc) - 1.0


def _z_ra(smiles: str) -> float:
    return 0.29056 - 0.08775 * _acentric(smiles)


from chemicals.volume import Yamada_Gunn


def _vs_cm3_mol(
    smiles: str, T: float = 288.15, use_thermo=False
) -> float:  # default 15C
    tc, pc = _tc(smiles), _pc(smiles)
    if use_thermo:
        vs = Yamada_Gunn(T, tc, pc * 1e5, _acentric(smiles))  # bar -> Pa
        return vs * 1e6  # m3 -> cm3
    if tc <= 0 or pc <= 0:
        return 0.0
    exponent = 1.0 + (1.0 - T / tc) ** (2.0 / 7.0)
    return (_R_BAR_CM3 * tc / pc) * (_z_ra(smiles) ** exponent)


def _density_g_cm3(smiles: str, T: float = 298.15) -> float:
    """Liquid density at T (default 15C)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0
    vs = _vs_cm3_mol(smiles, T)
    if vs <= 0.0:
        return 0.0
    mw = Descriptors.MolWt(mol)  # g mol-1
    return mw / vs


# -------------------------------------------------------------------------
# 4 NHOC adn density helpers


def _hf(smiles: str) -> float:
    """Standard enthalpy of formation dHf (kJ mol‑1, 298 K)."""
    return _group_contribution(smiles, _IDX_HF) + _HF0


def _nhoc_raw(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0
    mw_g_mol = Descriptors.MolWt(mol)
    if mw_g_mol == 0:  # RDKit kept only "dummy" atoms
        return 0.0
    nC, nH = _count_atoms(mol)
    delta_h_comb = nC * _DH_F_CO2 + (nH / 2.0) * _DH_F_H2O - _hf(smiles)
    return -delta_h_comb / mw_g_mol  # kJ g-1 = MJ kg-1


# -------------------------------------------------------------------------
# 5 Public call used by reward_fn


def nhoc(smiles: str) -> float:  # MJ kg-1
    return _nhoc_raw(smiles)


def vol_nhoc(smiles: str, T: float = 288.15) -> float:  # MJ L-1 at 15C by default
    rho = _density_g_cm3(smiles, T)
    return _nhoc_raw(smiles) * rho  # MJ kg-1 * g cm-3 -> MJ L-1


# -------------------------------------------------------------------------
# Normalisation and batch processing

# --- Volumetric NHOC (MJ L-1) ----------------------------------------
_VNHOC_MU = 34.0  # training mean
_VNHOC_SIGMA = 5.0  # training std
_VNHOC_Z_CLIP = 3.0  # hard clip for outliers/invalids
_VNHOC_TARGET = 37.0  # centre of "sweet-spot" window
_VNHOC_K = 1.3  # logistic slope  (ca 1 / sigma)


def _scale_vnhoc_clipped_logistic(x: float) -> float:
    """
    1. z-score w.r.t. training distribution
    2. clip to +-3 sigma to tame invalid 0-MJ points
    3. logistic desirability centred at the aviation-fuel target
    """
    z = (x - _VNHOC_MU) / _VNHOC_SIGMA
    z = max(min(z, _VNHOC_Z_CLIP), -_VNHOC_Z_CLIP)
    z0 = (_VNHOC_TARGET - _VNHOC_MU) / _VNHOC_SIGMA
    return 1.0 / (1.0 + math.exp(-_VNHOC_K * (z - z0)))


def batch_vol_nhoc(smiles_list: Sequence[str], T: float = 288.15) -> List[float]:
    """
    Volumetric NHOC reward in [0, 1].
    Invalid SMILES or NHOC < 5 MJ L-1 receive 0.
    """
    rewards: List[float] = []
    for sm in smiles_list:
        mol_ok = Chem.MolFromSmiles(sm) is not None
        if not mol_ok:
            rewards.append(0.0)
            continue
        try:
            vn = vol_nhoc(sm, T)
        except Exception:
            rewards.append(0.0)
            continue
        if vn < 5.0:  # trap truly bad estimates
            rewards.append(0.0)
        else:
            rewards.append(_scale_vnhoc_clipped_logistic(vn))
    return rewards


# --- Gravimetric NHOC (MJ kg-1) --------------------------------------
NHOC_MASS_LB = 40.0  # 0-reward below this
NHOC_MASS_TAR = 44.0  # 1-reward at / above this


def desirability_nhoc(value: float, s: float = 1.0) -> float:
    if value <= NHOC_MASS_LB:
        return 0.0
    if value >= NHOC_MASS_TAR:
        return 1.0
    return ((value - NHOC_MASS_LB) / (NHOC_MASS_TAR - NHOC_MASS_LB)) ** s


def batch_nhoc(smiles_list: Sequence[str]) -> List[float]:
    rewards: List[float] = []
    for sm in smiles_list:
        mol_ok = Chem.MolFromSmiles(sm) is not None
        if not mol_ok:
            rewards.append(0.0)
            continue
        rewards.append(desirability_nhoc(nhoc(sm)))
    return rewards
