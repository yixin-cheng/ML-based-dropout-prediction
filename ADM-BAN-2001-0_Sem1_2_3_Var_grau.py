
# coding: utf-8

# # Time Series Analysis PUC Data

# In[1]:


import pandas as pd
import numpy as np
import datetime
from collections import Counter

#ML Libraries
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn import preprocessing
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

#Feature extraction
from tsfresh import extract_relevant_features


# ## Functions

# In[19]:



def cleannanvalues(df, datos_series, method = 'zeros'):
    """Clean NAN Values.

    Clean NAN values after the data frame creation.
    Normalize between 0-1, selected variables (i.e. only those that will be used for time series modeling).

    Parameters
    ----------
    df: pandas dataframe
    datos_series : list iterable
        Columns names in df that will be clean.
    method : str 
        Zeros: Replace NAN values per zeros.
        TODO: Hace falta programas otras opciones
    
    Returns
    -------
    df dataframe
    """
    for label in datos_serie:
        # NaN strategy
        if method == 'zeros':
            df[label].fillna(0, inplace=True)
        #Normalization
        df[label] = (df[label]-df[label].min())/(df[label].max()-df[label].min())
    return df

def crea_disciplinas_series_index(df_curso, tipo = 1, min_samples = 20):
    """ Creates the dictionary of disciplines that will be used to create the series. Column names 
    Parameters
    ----------
    df_curso: pandas dataframe
    tipo : int 
        TIPO 1: Todas las disciplinas que tengan al menos min_samples en el semestre seleccionado. 
        TIPO 2: Solo las materias que son recomendadas TODO.
    min_samples : int > 0 or -1 
        Minimun number of samples of echa course per semestre to be included.
        If tou want to include all possible courses in dataset set -1
    Returns
    -------
    list of selected disciplinees per semester
    """
    # TODO: Hay que probar diferentes formas de crear la serie
    disciplinas_series_index = {} 
    for semestre in semestre_do_aluno_model:
        temp_semes = df_to_createseries.loc[df_curso['semestre_do_aluno'] == semestre]
        if tipo == 1:
            tem_min = temp_semes.groupby("grupos").filter(lambda x: len(x) > min_samples)
            disciplinas_series_index[semestre] = sorted(tem_min['grupos'].unique().tolist())
        elif tipo == 2:
            #TODO: Hacer esta implementacion
            disciplinas_series_index[semestre] = sorted(temp['grupos'].unique().tolist())
    return disciplinas_series_index

def crea_dict_por_matricula(disciplinas_series_index, matricula_serie, matricula, missing_discipline = 0):
    """ Recibe a student (matricula) and creates a list of dictionaries. Each dictionary represent the data of one semester.
    
    Parameters
    ----------
    disciplinas_series_index: list iterable
        Contains the set of discipline per semester to be consider.
    matricula_serie : dict 
        Contains the student organized per semeter
    matricula : str
        Student id
    missing_discipline: int  (0 default)
        Missing discipline == When a student is not enrolled in a particular selected course.
        Whe the is a missing discipline the vairables values are replace by the value specified by this parameter.
    
    Returns
    -------
    list of dictionaries
    lenght of the list == number of the semesters considered
    
    """
    if len(semestre_do_aluno_model) != len(matricula_serie): #Hay data de la matricula en los semestres considerados
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
                        print matricula_serie[semestre][disciplina]
                        raise ValueError('Valor de un dato de serie errado. Revisar matricula: ' + matricula_serie[semestre][disciplina]['matricula'] + ' Semestre: '+ str(semestre) + ' Disciplina: '+ str(disciplina))

            else: #Missing Data
                for dato in datos_serie:
                    key = disciplina + "_" + dato
                    serie[key] = missing_discipline
        #print serie
        #print len(serie)
        if len(serie) != ((len_total*len(datos_serie))+2):#+2 por el indice del id y del semestre
            raise ValueError('La longitud esperada de la serie no se obtuvo!!!')
        
        series_total.append(serie)
        
    #print len(series_total) #=Numero de Semestres
    #print series_total
    return series_total

def create_series_and_labels_course_variable(df_to_createseries,disciplinas_series_index):
    """
    This function is quite important!!, creates the series and the serieslabels. Each course is a time series variable.
    
    Parameters
    ----------
    df_to_createseries : data_frame
        Este dataframe se va a agrupar por matricula.
    disciplinas_series_index : dict 
        Disciplinas que va a ser consideradas

    Returns
    -------
    series_model: list of series
    label_model: list of series laberls
    
    """
    series_model = [] # Series de los alumnos seleccionados
    label_model = {} # Label : Dropout/No_Dropout

    for matricula,df_aluno in df_to_createseries.groupby('matricula'):
        gp_sems = df_aluno.groupby('semestre_do_aluno')
        matricula_serie = {}
        for semestre, df_sem_al in gp_sems:
            matricula_serie[semestre] = {}
            for index, row in df_sem_al.iterrows():
                matricula_serie[semestre][row['grupos']] =  row        

        label_sit = list((df_aluno.sit_vinculo_atual.tolist()))[0]

        serie_matricula = crea_dict_por_matricula(disciplinas_series_index,matricula_serie,matricula,missing_discipline = missing_disc)

        if serie_matricula: #Existe el caso que ninguna de las disciplinas en disciplinas_series_index haya sido inscrita!!
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



dtype = {'ano_curriculo' : np.string_ ,'cod_curriculo' : np.string_ , 'mat_ano' : np.int8, 'mat_sem' : np.int8, 'periodo' : np.string_ , 
         'ano' : np.string_ , 'semestre' : np.int8, 'semestre_recomendado' : np.int8, 'semestre_do_aluno' : np.int8, 'no_creditos' : np.int8,
         'cep' :  np.string_ , 'puntos_enem' : np.float32 , 'diff' : np.int8 , 'tentativas' : np.int8, 'cant' : np.int8,
        'identificador': np.string_, 'cod_curriculo': np.int8, 'cod_enfase' : np.string_}


dfh = pd.read_csv('historicosFinal.csv', sep=';', dtype = dtype, converters={'grau': lambda x: x.replace(',','.')} )
#Problem reading floats => dataset with , not .
dfh = dfh.applymap(lambda x: x.strip() if type(x) is str else x)
dfh['grau'] = dfh['grau'].apply(pd.to_numeric)



# ## Data grouping and filtering
# 
# En esta seccion se escogen la carrera, los estados de dropout, las variables de la serie, los semestres a considerar en la construccion de la serie.

# In[23]:


# Arquitectura: ARQ-BAQ-2002-0
# Dereito CDD/CDD-BDD-CON-2008-0
# Computacion CSI-BID-2010-0 (39 dropout y 79 no dropout)
# Administracion: ADM-BAN-2001-0
# Datos de filtrado
cod_curso = 'ADM'
identificador = 'ADM-BAN-2001-0'
cod_curriculo = 0

#Datos de construccion de la serie
semestre_do_aluno_model = [1,2,3] #Semestres usados para modelar
drop_out_labels = ['DESLIGADO','JUBILADO','MATRICULA EM ABANDONO'] # Estados considerados como dropout
#datos_serie = ['grau','diff','tentativas'] #Que datos de una disciplina se van a incluir en la serie?
datos_serie = ['grau'] 
missing_disc = -1 #Valor en la serie cuando no se inscribe un curso
name_file = identificador + "_Sem" + '_'.join(str(e) for e in semestre_do_aluno_model) + "_Var_" + '_'.join(datos_serie)
print "File: " + name_file

cleannanvalues(dfh,datos_serie)
df_curso = dfh.groupby(['cod_curso','cod_curriculo','identificador'])
df_curso= df_curso.get_group((cod_curso,cod_curriculo,identificador))


df_to_createseries = df_curso.loc[df_curso['semestre_do_aluno'].isin(semestre_do_aluno_model)]
#df_to_createseries.describe()



# ## Series creation

# In[11]:



disciplinas_series_index = crea_disciplinas_series_index(df_curso, tipo = 1, min_samples = 20)
print "Disciplinas used to build the series:"
print disciplinas_series_index

series_model, label_model = create_series_and_labels_course_variable(df_to_createseries,disciplinas_series_index)

print "Number of rows in final series (Matriculas * Num Semestres): "
print len(series_model) #Longitud de la serie
print "Ejemplo de Serie Indice 1: "
print series_model[0]
print "Number of Matriculas: "
print len(label_model)# Numero de Matricula



# In[10]:


timeseries =pd.DataFrame(series_model)
timeseries_label = pd.Series(label_model)

#NAN -> cuando se pasa a dataframe no existe data de la serie para cursos que nadie vio en x semestre pero si en y para 0<=x<y
#Interpretacion=> no vio el curso en x reemplazarlo por missing_discipline
timeseries = timeseries.fillna(missing_disc)


timeseries.isnull().any()

timeseries.to_csv(name_file+".csv", sep=';')
features_filtered_direct = extract_relevant_features(timeseries, timeseries_label, column_id='id', column_sort='semestre')
features_filtered_direct.to_csv(name_file+"_features.csv", sep=';')
timeseries_label.to_csv(name_file+"_labels.csv", sep=';')


# ## Machine Learning Training Models

# In[17]:



#Hay que comentar name_file para una corrida completa.
#name_file = "TimeSeriesProgramsToRunServer/Res_Feature_Extraction/ARQ-BAQ-2002-0_Sem1_2_3_Var_grau"

X = pd.read_csv(name_file + '_features.csv', sep=';' )
y_read = pd.read_csv(name_file + '_labels.csv', sep=';', header = None , names = ['id','class'], index_col = 'id' ).to_dict('index')

y = []
for alumno in X['id']:
    y.append(y_read[alumno]['class'])

X_selected = X.drop(['id'], axis=1)

print "Dimensiones antes del sampling: "
print Counter(y).items() #Antes del Sampling
#X_resampled, y_resampled = RandomOverSampler(random_state=42).fit_sample(X_selected, y)
X_resampled, y_resampled = SMOTE(random_state=42).fit_sample(X_selected, y)
print "Dimensiones despues del sampling: "
print Counter(y_resampled).items() #Despues del Sampling

#X_selected.head()

scoring = {'accuracy': 'accuracy',
           'precision': 'precision',
           'recall': 'recall',
           'f1': 'f1'}


clasificadores_score = {}

#dt = DecisionTreeClassifier()
#scores = cross_validate(dt, X_resampled, y_resampled, scoring=scoring, cv=10)
#res_tem = {"Acc" : np.average(scores['test_accuracy']), "Recall": np.average(scores['test_recall']),
#          "Precision": np.average(scores['test_precision']), "F1": np.average(scores['test_f1'])}
#clasificadores_score['DecisionTreeClassifier'] = res_tem


rf = RandomForestClassifier(n_estimators = 200)
scores = cross_validate(rf, X_resampled, y_resampled, scoring=scoring, cv=10)
res_tem = {"Acc" : np.average(scores['test_accuracy']), "Recall": np.average(scores['test_recall']),
          "Precision": np.average(scores['test_precision']), "F1": np.average(scores['test_f1'])}
clasificadores_score['RandomForestClassifier'] = res_tem


gbc = GradientBoostingClassifier(learning_rate=0.1,n_estimators=200,max_depth=10)
scores = cross_validate(gbc, X_resampled, y_resampled, scoring=scoring, cv=10)
res_tem = {"Acc" : np.average(scores['test_accuracy']), "Recall": np.average(scores['test_recall']),
          "Precision": np.average(scores['test_precision']), "F1": np.average(scores['test_f1'])}
clasificadores_score['GradientBoostingClassifier'] = res_tem

gnb = GaussianNB()
scores = cross_validate(gnb, X_resampled, y_resampled, scoring=scoring, cv=10)
res_tem = {"Acc" : np.average(scores['test_accuracy']), "Recall": np.average(scores['test_recall']),
          "Precision": np.average(scores['test_precision']), "F1": np.average(scores['test_f1'])}
clasificadores_score['GaussianNB'] = res_tem

svc = SVC(C=1)
scores = cross_validate(svc, X_resampled, y_resampled, scoring=scoring, cv=10)
res_tem = {"Acc" : np.average(scores['test_accuracy']), "Recall": np.average(scores['test_recall']),
          "Precision": np.average(scores['test_precision']), "F1": np.average(scores['test_f1'])}
clasificadores_score['SVC'] = res_tem


lr = LogisticRegression()
scores = cross_validate(lr, X_resampled, y_resampled, scoring=scoring, cv=10)
res_tem = {"Acc" : np.average(scores['test_accuracy']), "Recall": np.average(scores['test_recall']),
          "Precision": np.average(scores['test_precision']), "F1": np.average(scores['test_f1'])}
clasificadores_score['lr'] = res_tem

#xgboost?

print "RESULTADOS DE LOS MODELOS DE CLASIFICACION"
print clasificadores_score


