import streamlit as st
import pandas as pd
import subprocess
from PIL import Image
import os
import base64
import pickle
import sklearn


# Molecular descriptor calculator
def desc_calc():
    # Performs the descriptor calculation
    bashCommand = "java -Xms2G -Xmx2G -Djava.awt.headless=true -jar ./PaDEL-Descriptor/PaDEL-Descriptor.jar -removesalt -standardizenitro -fingerprints -descriptortypes ./PaDEL-Descriptor/%s -dir ./ -file descriptors_output.csv" % selected_fp 
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

    # Read in calculated descriptors and display the dataframe
    st.subheader('Calculated molecular descriptors')
    desc = pd.read_csv('descriptors_output.csv')
    st.write(desc)
    st.markdown(filedownload(desc), unsafe_allow_html=True)
    # Write the data dimension (number of molecules and descriptors)
    nmol = desc.shape[0]
    ndesc = desc.shape[1]
    st.info('Selected fingerprint: ' + user_fp)
    st.info('Number of molecules: ' + str(nmol))
    st.info('Number of descriptors: ' + str(ndesc-1))
    os.remove('molecule.smi')

# File download
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="descriptor_{user_fp}.csv">Download CSV File</a>'
    return href


def filedownload(df1):
    csv = df1.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download Predictions</a>'
    return href


# Logo image
image = Image.open('Applogo.png')

st.image(image, use_column_width=True)
# Page title
st.markdown("""
# MolDesc-BioPredictor : Molecular Descriptor Calculator and Bioactivity Prediction App 
This app allows you to calculate descriptors of molecules ( **molecular descriptors**) that you can use for computational drug discovery projects such as for the construction of quantitative structure-activity/property relationship (QSAR/QSPR) models.This app allows you to predict the bioactivity towards inhibting the enzyme which is a drug target for disease.

In this app we can calculate 12 **molecular fingerprints** (`PubChem`, `AtomPairs2D`, `AtomPairs2DCount`, `CDK`, `CDKextended`, `CDKgraphonly`, `EState`, `KlekotaRoth`, `KlekotaRothCount`, `MACCS`, `Substructure` and `SubstructureCount`).

**Credits**
- App built in `Python` + `Streamlit` 
- Descriptor calculated using [PaDEL-Descriptor](http://www.yapcwsoft.com/dd/padeldescriptor/) software.
- Yap CW. [PaDEL‐descriptor: An open source software to calculate molecular descriptors and fingerprints](https://doi.org/10.1002/jcc.21707). ***J Comput Chem*** 32 (2011) 1466-1474.
---
""")

# Sidebar
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/Sayalitawade/ML_project/main/Amyloid_pInhibition.csv)
""")

with st.sidebar.header('2. Enter column names for 1) Molecule ID and 2) SMILES'):
    name_mol = st.sidebar.text_input('Enter column name for Molecule ID', 'molecule_chembl_id')
    name_smiles = st.sidebar.text_input('Enter column name for SMILES', 'canonical_smiles')

with st.sidebar.header('3. Set parameters'):
    # Select fingerprint
    fp_dict = {'PubChem':'PubchemFingerprinter.xml',
               'AtomPairs2D':'AtomPairs2DFingerprinter.xml',
               'AtomPairs2DCount':'AtomPairs2DFingerprintCount.xml',
               'CDK':'Fingerprinter.xml',
               'CDKextended':'ExtendedFingerprinter.xml',
               'CDKgraphonly':'GraphOnlyFingerprinter.xml',
               'EState':'EStateFingerprinter.xml',
               'KlekotaRoth':'KlekotaRothFingerprinter.xml',
               'KlekotaRothCount':'KlekotaRothFingerprintCount.xml',
               'MACCS':'MACCSFingerprinter.xml',
               'Substructure':'SubstructureFingerprinter.xml',
               'SubstructureCount':'SubstructureFingerprintCount.xml'}
    user_fp = st.sidebar.selectbox('Choose fingerprint to calculate', list(fp_dict.keys()))
    selected_fp = fp_dict[user_fp]

    # Set number of molecules to compute
    df0 = pd.read_csv('Amyloid_pInhibition.csv')
    all_mol = df0.shape[0]
    number2calc = st.sidebar.slider('How many molecules to compute?', min_value=10, max_value=all_mol, value=10, step=10)

if uploaded_file is not None:
    # Read CSV data
    @st.cache
    def load_csv():
        csv = pd.read_csv(uploaded_file).iloc[:number2calc, 1:]
        return csv
    df = load_csv()
    df2 = pd.concat([df[name_smiles], df[name_mol]], axis=1)
    # Write CSV data
    df2.to_csv('molecule.smi', sep='\t', header=False, index=False)
    st.subheader('Initial data from CSV file')
    st.write(df)
    st.subheader('Formatted as PADEL input file')
    st.write(df2)
    with st.spinner("Calculating descriptors..."):
        desc_calc() 


# Model building
def build_model(input_data):

    # Reads in saved regression model
    load_model = pickle.load(open('ML_model.pkl', 'rb'))
    # Apply model to make predictions
    prediction = load_model.predict(input_data)
    st.header('**Prediction output**')
    prediction_output = pd.Series(prediction, name='pActivity')
    molecule_name = pd.Series(load_data[0], name='molecule_name')
    df1 = pd.concat([molecule_name, prediction_output], axis=1)
    st.write(df1)
    st.markdown(filedownload(df1), unsafe_allow_html=True)


if st.sidebar.button('Predict Bioactivity'):

    if selected_fp == fp_dict['PubChem']:
        load_data = pd.read_table(uploaded_file, sep=' ', header=None)
        load_data.to_csv('molecule.smi', sep='\t', header=False, index=False)

        # Read in calculated descriptors and display the dataframe
        st.header('**Calculated PubChem descriptors**')
        desc = pd.read_csv('descriptors_output.csv')
        st.write(desc)
        st.write(desc.shape)

        # Read descriptor list used in previously built model
        st.header('**Subset of descriptors from previously built models**')
        Xlist = list(pd.read_csv('descriptor_list.csv').columns)
        desc_subset = desc[Xlist]
        st.write(desc_subset)
        st.write(desc_subset.shape)

        # Apply trained model to make prediction on query compounds
        build_model(desc_subset)
    else:
        st.info('Select PubChem fingerprint from sidebar for Bioactivity prediction')

else:
    st.info('Click Predict button to start!')


