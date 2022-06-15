
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import datetime
from collections import Counter
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.cuda
#ML Libraries
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn import preprocessing
from sklearn.model_selection import cross_validate

from feature_selector import Selector
from preprocessor import Preprocessor
#Feature extraction
from tsfresh import extract_relevant_features


# ## Functions

# In[19]:
if torch.cuda.is_available():
    device = torch.device("cuda:2")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")


def cleannanvalues(df, datos_serie, method):
    """Clean NAN Values.

    Clean NAN values after the data frame creation.
    Normalize between 0-1, selected variables (i.e. only those that will be used for time series modeling).

    Parameters
    ----------
    df: pandas dataframe
    datos_serie : list iterable
        Columns names in df that will be clean.
    method : str 
        Zeros: Replace NAN values per zeros.
        TODO: There are other optional programs (change 0 to some predicted values)

    Returns
    -------
    df dataframe
    """
    # Preprocessor(initial_path, target_path)
    # df_target_path=pd.read_csv(target_path)
    for label in datos_serie:
        # NaN strategy
        if method == 'zeros':
            df[label].fillna(0, inplace=True)
        elif method=='impute':
            if label=='grau':
                df[label] = df_target_path[label]
            else:
                df[label].fillna(0, inplace=True)

        #Normalization
        df[label] = (df[label]-df[label].min())/(df[label].max()-df[label].min())
    return df

def crea_disciplinas_series_index(df_curso, type = 1, min_samples = 20):
    """ Creates the dictionary of disciplines that will be used to create the series. Column names 
    Parameters
    ----------
    df_curso: pandas dataframe
    tipo : int 
        TYPE 1: All subjects that have at least min_samples in the selected semester.
       TYPE 2: Solo the materials that are recommended  TODO.
    min_samples : int > 0 or -1 
        Minimum number of samples of echa course per semester to be included.
        If tou want to include all possible courses in dataset set -1
    Returns
    -------
    list of selected disciplinees per semester
    """
    # TODO: There are different ways to create the series
    disciplinas_series_index = {} 
    for semestre in semestre_do_aluno_model:
        #df_to_createseries = df_curso.loc[df_curso['semestre_do_aluno'].isin(semestre_do_aluno_model)]
        temp_semes = df_to_createseries.loc[df_curso['semestre_do_aluno'] == semestre]
        if type == 1:
            tem_min = temp_semes.groupby("grupos").filter(lambda x: len(x) > min_samples)
            disciplinas_series_index[semestre] = sorted(tem_min['grupos'].unique().tolist())
        elif type == 2:
            pass
            #TODO: Make this implementation
            # disciplinas_series_index[semestre] = sorted(temp['grupos'].unique().tolist())
    return disciplinas_series_index

def crea_dict_por_matricula(disciplinas_series_index, matricula_serie, matricula, missing_discipline = 0):
    """ Receive a student (enrollment) and creates a list of dictionaries. Each dictionary represent the data of one semester.
    
    Parameters
    ----------
    disciplinas_series_index: list iterable
        Contains the set of discipline per semester to be consider.
    matricula_serie : dict 
        Contains the student organized per semester
    matricula : str
        Student id
    missing_discipline: int  (0 default)
        Missing discipline == When a student is not enrolled in a particular selected course.
        Whe the is a missing discipline the vairables values are replace by the value specified by this parameter.
    
    Returns
    -------
    list of dictionaries
    length of the list == number of the semesters considered
    
    """
    if len(semestre_do_aluno_model) != len(matricula_serie): #There is enrollment data in the semesters considered
        return None
    series_total = []
    len_total = 0
    for semestre in semestre_do_aluno_model:
        len_total = len(disciplinas_series_index[semestre])
        serie = {}
        serie['semestre'] = semestre
        serie['id'] = matricula

        for disciplina in disciplinas_series_index[semestre]:
            if disciplina in matricula_serie[semestre]:
                #Vio la disciplina
                for dato in datos_serie:
                    if matricula_serie[semestre][disciplina][dato] is not None and not np.isnan(matricula_serie[semestre][disciplina][dato]):
                        key = disciplina + "_" + dato
                        serie[key] = matricula_serie[semestre][disciplina][dato]
                    else:
                        print (matricula_serie[semestre][disciplina])
                        raise ValueError('Value of a wrong series data. Check enrollment: ' + matricula_serie[semestre][disciplina]['matricula'] + ' Semestre: '+ str(semestre) + ' Disciplina: '+ str(disciplina))

            else: #Missing Data
                for dato in datos_serie:
                    key = disciplina + "_" + dato
                    serie[key] = missing_discipline
        #print serie
        #print len(serie)
        if len(serie) != ((len_total*len(datos_serie))+2):#+2 for the ID and semester index
            raise ValueError('La longitud esperada de la serie no se obtuvo!!!')
        
        series_total.append(serie)

    #print len(series_total) #=Numero de Semestres
    # print ('series_total',series_total)
    return series_total

def create_series_and_labels_course_variable(df_to_createseries,disciplinas_series_index):
    """
    This function is quite important!!, creates the series and the series labels. Each course is a time series variable.
    
    Parameters
    ----------
    df_to_createseries : data_frame
        Este dataframe se va a agrupar por matricula.
        This dataframe is going to be grouped by license.
    disciplinas_series_index : dict 
        Disciplines to be considered

    Returns
    -------
    series_model: list of series
    label_model: list of series labels
    
    """
    series_model = [] # Series of selected students
    label_model = {} # Label : Dropout/No_Dropout

    for matricula,df_aluno in df_to_createseries.groupby('matricula'):
        gp_sems = df_aluno.groupby('semestre_do_aluno')
        matricula_serie = {}
        for semestre, df_sem_al in gp_sems:
            matricula_serie[semestre] = {}
            for index, row in df_sem_al.iterrows():
                matricula_serie[semestre][row['grupos']] =  row        

        label_sit = list((df_aluno.sit_vinculo_atual.tolist()))[0]
        # print('matricula_serie: ', matricula_serie)
        serie_matricula = crea_dict_por_matricula(disciplinas_series_index,matricula_serie,matricula,missing_discipline = missing_disc)
        # print('serie_matricula:??? ',serie_matricula)
        if serie_matricula: #There is the case that none of the disciplines in disciplines_series_index has been registered !!
            series_model.extend(serie_matricula)

            if label_sit in drop_out_labels:
                label_model[matricula] = 1  #DROPOUT Class Label
            else:
                label_model[matricula] = 0 #DROPOUT Class Label
        else:
            continue
    return series_model, label_model


# ## CSV read and Dataset main fields explanation
#     - tentativas : attemps
#     - diff : difference between the recommended semester for a discipline and the actual student semester
#     - puntos_enem : admision score?
#     - matricula : student id
#     - semestre_do_aluno : current student semester
#     - sit_final : pass (AP), not pass (RP)
#     - sit_vinculo_actual : [JUBILADO, DESLIGADO, MATRICULA EM ABANDONO,...] 

# In[18]:

initial_path='data/historicosFinal.csv'

dtype = {'ano_curriculo' : np.string_ ,'cod_curriculo' : np.string_ , 'mat_ano' : np.int8, 'mat_sem' : np.int8, 'periodo' : np.string_ , 
         'ano' : np.string_ , 'semestre' : np.int8, 'semestre_recomendado' : np.int8, 'semestre_do_aluno' : np.int8, 'no_creditos' : np.int8,
         'cep' :  np.string_ , 'puntos_enem' : np.float32 , 'diff' : np.int8 , 'tentativas' : np.int8, 'cant' : np.int8,
        'identificador': np.string_, 'cod_curriculo': np.int8, 'cod_enfase' : np.string_}


dfh = pd.read_csv(initial_path, sep=';', dtype = dtype, converters={'grau': lambda x: x.replace(',','.')} )
#Problem reading floats => dataset with , not .
dfh = dfh.applymap(lambda x: x.strip() if type(x) is str else x) # remove the beginning and end space of string, apply it in each element
# dfh.to_csv('111.csv',index=False)
dfh['grau'] = dfh['grau'].apply(pd.to_numeric)



# ## Data grouping and filtering
# 
# In this section, the career, the dropout states, the variables of the series, the semesters to be considered in the construction of the series are chosen.

# In[24]:


# Arquitectura: ARQ-BAQ-2002-0
# Dereito CDD/CDD-BDD-CON-2008-0
# Computacion CSI-BID-2010-0 (39 dropout y 79 no dropout)
# Administracion: ADM-BAN-2001-0
# Datos de filtrado
cod_curso = 'ADM' #code of course
identificador = 'ADM-BAN-2001-0'
cod_curriculo = 0 # code of curriculum

#Series construction data
semestre_do_aluno_model = [1,2] #Semesters used for modeling
drop_out_labels = ['DESLIGADO','JUBILADO','MATRICULA EM ABANDONO'] # States considered as dropout
#datos_serie = ['grau','diff','tentativas'] #What data of a discipline will be included in the series?
datos_serie = ['grau'] 
missing_disc = -1   #Value in series when no course is enrolled
name_file = identificador + "_Sem" + '_'.join(str(e) for e in semestre_do_aluno_model) + "_Var_" + '_'.join(datos_serie)
print ("File: " + name_file)

cleannanvalues(dfh, datos_serie,'zeros')
df_curso = dfh.groupby(['cod_curso','cod_curriculo','identificador'])
df_curso= df_curso.get_group((cod_curso,cod_curriculo,identificador))


# select the desired semester to proceed
df_to_createseries = df_curso.loc[df_curso['semestre_do_aluno'].isin(semestre_do_aluno_model)]



# ## Series creation

# In[11]:



disciplinas_series_index = crea_disciplinas_series_index(df_curso, type = 1, min_samples = 20)
print ("Disciplinas used to build the series:")
print (disciplinas_series_index)

series_model, label_model = create_series_and_labels_course_variable(df_to_createseries,disciplinas_series_index)

print ("Number of rows in final series (Matriculas * Num Semestres): ")
print (len(series_model)) #series length
print ("Index Series Example 1: ")
# print (series_model[0])
# print(series_model[1])
print ("Number of Matriculas: ")
print (len(label_model))# Enrollment Number
print (label_model)



# In[10]:


# timeseries =pd.DataFrame(series_model)
# timeseries_label = pd.Series(label_model)
#
# #NAN -> when it is passed to dataframe there is no data from the series for courses that nobody saw in x semester but in y for 0 <= x <y
# #Interpretacion=> did not see the course in x replace it with missing_discipline
# timeseries = timeseries.fillna(missing_disc)
# #
# #
# timeseries.isnull().any()
# #
# timeseries.to_csv(name_file+".csv", sep=';')
# features_filtered_direct = extract_relevant_features(timeseries, timeseries_label, column_id='id', column_sort='semestre')
# features_filtered_direct.to_csv(name_file+"_features.csv", sep=';')
# timeseries_label.to_csv(name_file+"_labels.csv", sep=';')


# ## Machine Learning Training Models

# In[26]:



#Name_file must be commented for a full run.
#name_file = "TimeSeriesProgramsToRunServer/Res_Feature_Extraction/ADM-BAN-2001-0_Sem1_2_3_4_Var_grau"

X = pd.read_csv(name_file + '_features.csv', sep=';' )
y_read = pd.read_csv(name_file + '_labels.csv', sep=';', header = None , names = ['id','class'], index_col = 'id' ).to_dict('index')

y = []
for alumno in X['id']:
    y.append(y_read[alumno]['class'])

X_selected = X.drop(['id'], axis=1)

# print ("Dimensiones antes del sampling: ")
# print (Counter(y).items()) #Before Sampling
# #X_resampled, y_resampled = RandomOverSampler(random_state=42).fit_sample(X_selected, y)
# X_resampled, y_resampled = SMOTE(random_state=42).fit_resample(X_selected, y)
# print ("Dimensiones despues del sampling: ")
# print (Counter(y_resampled).items()) #After Sampling
data=pd.concat([X_selected.assign(drop=i) for i in y],ignore_index=False)

# data.to_csv(target_path,header=True,index=True)
print('getting into feature selection---')
data=Selector(data)  # apply GA for feature selection
print('after')
if torch.cuda.is_available():
    device = torch.device("cuda:2")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

# Hyper Parameters
input_size = len(data.columns) - 1
hidden_size = 50
num_classes = 2
num_epochs = 500
batch_size = 10
learning_rate = 0.01


# define a function to plot confusion matrix
def plot_confusion(input_sample, num_classes, des_output, actual_output):
    confusion = torch.zeros(num_classes, num_classes)
    for i in range(input_sample):
        actual_class = actual_output[i]
        predicted_class = des_output[i]

        confusion[actual_class][predicted_class] += 1

    return confusion


"""
Step 1: Load data and pre-process data
Here we use data loader to read data
"""


# define a customise torch dataset
class DataFrameDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.data_tensor = torch.Tensor(df.values)

    # a function to get items by index
    def __getitem__(self, index):
        obj = self.data_tensor[index]
        input = self.data_tensor[index][0:-1]
        target = self.data_tensor[index][-1]

        return input, target

    # a function to count samples
    def __len__(self):
        n, _ = self.data_tensor.shape
        return n


# load all data
# data = pd.read_csv(target_path, header=None, index_col=0)

# normalise input data
# for column in data.columns[:-1]:
#     # the last column is target
#     data[column] = data.loc[:, [column]].apply(lambda x: (x - x.mean()) / x.std())

# randomly split data into training set (80%) and testing set (20%)
msk = np.random.rand(len(data)) < 0.8
train_data = data[msk]
test_data = data[~msk]

# define train dataset and a data loader
train_dataset = DataFrameDataset(df=train_data)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

"""
Step 2: Define a neural network 

Here we build a neural network with one hidden layer.
    input layer: n neurons, representing the features of status of students
    hidden layer: 50 neurons, using Sigmoid/Tanh/Tanh/Softmax as activation function

    output layer: 2 neurons, representing the type of dropouts
"""


# Neural Network
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.Tanh = nn.Tanh()
        self.Softmax = nn.Softmax()
        self.Relu=nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.sigmoid(out)
        out = self.Tanh(out)
        out = self.Softmax(out)
        out = self.Relu(out)
        out = self.fc2(out)
        return out


net = Net(input_size, hidden_size, num_classes).to(device=device)

# Loss and Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate)

# store all losses for visualisation
all_losses = []

# train the model by batch
for epoch in range(num_epochs):
    total = 0
    correct = 0
    total_loss = 0
    for step, (batch_x, batch_y) in enumerate(train_loader):
        X = batch_x.to(device)
        Y = batch_y.long().to(device)
        # print(X)
        # Forward + Backward + Optimize
        optimizer.zero_grad()  # zero the gradient buffer
        outputs = net(X)
        loss = criterion(outputs, Y)
        all_losses.append(loss.item())
        loss.backward()
        optimizer.step()

        if (epoch % 50 == 0):
            _, predicted = torch.max(outputs, 1)
            # calculate and print accuracy
            total = total + predicted.size(0)
            correct = correct + sum(predicted.data.numpy() == Y.data.numpy())
            total_loss = total_loss + loss
    if (epoch % 50 == 0):
        print('Epoch [%d/%d], Loss: %.4f, Accuracy: %.2f %%'
              % (epoch + 1, num_epochs,
                 total_loss, 100 * correct / total))

# Optional: plotting historical loss from ``all_losses`` during network learning
# Please uncomment me from next line to ``plt.show()`` if you want to plot loss

# import matplotlib.pyplot as plt
#
# plt.figure()
# plt.plot(all_losses)
# plt.show()

"""
Evaluating the Results


"""

train_input = train_data.iloc[:, :input_size]
train_target = train_data.iloc[:, input_size]

inputs = torch.Tensor(train_input.values).float()
targets = torch.Tensor(train_target.values - 1).long()

outputs = net(inputs)
_, predicted = torch.max(outputs, 1)

print('Confusion matrix for training:')
print(plot_confusion(train_input.shape[0], num_classes, predicted.long().data, targets.data))

"""
Step 3: Test the neural network

Pass testing data to the built neural network and get its performance
"""
# get testing data
test_input = test_data.iloc[:, :input_size]
test_target = test_data.iloc[:, input_size]

inputs = torch.Tensor(test_input.values).float()
targets = torch.Tensor(test_target.values - 1).long()

outputs = net(inputs)
_, predicted = torch.max(outputs, 1)

total = predicted.size(0)
correct = predicted.data.numpy() == targets.data.numpy()

print('Testing Accuracy: %.2f %%' % (100 * sum(correct) / total))

"""
Evaluating the Results

To see how well the network performs on different categories, will
create a confusion matrix, indicating for every  (rows)
which class the network (columns).

"""

print('Confusion matrix for testing:')
print(plot_confusion(test_input.shape[0], num_classes, predicted.long().data, targets.data))