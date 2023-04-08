
import datetime as dt
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import json

import statsmodels.api as sm
import statsmodels.tsa.api as smt
from scipy import stats
from statsmodels.tsa.api import SARIMAX
import os
import calendar
from matplotlib.colors import SymLogNorm #, Normalize
from matplotlib.ticker import MaxNLocator

import pymc3 as pm
import theano
import theano.tensor as tt
import arviz as az



def saveDict(outdict,filename='/home/pi/MonitorStation/battery.json'):
    with open(filename, 'w') as outfile:
        json.dump(outdict, outfile,sort_keys=True,indent=4) # separators=(',', ': ')

def loadDict(filename='/home/pi/MonitorStation/battery.json'):
    with open(filename) as f:
        dictobj = json.load(f)
    return dictobj



def saveDict(outdict,filename=''):
    with open(filename, 'w') as outfile:
        json.dump(outdict, outfile,sort_keys=True,indent=4) # separators=(',', ': ')

def loadDict(filename=''):
    with open(filename) as f:
        dictobj = json.load(f)
    return dictobj




def showRhat(trace):
    rhat = az.rhat(trace)
    fig, ax = plt.subplots(figsize=(15, 8))
    ax = (rhat.max().to_array().to_series().plot(kind="barh"))
    ax.axvline(1, c="k", ls="--");
    ax.set_xlabel(r"$\hat{R}$");
    ax.invert_yaxis();
    ax.set_ylabel(None);
    plt.show()


def showSummary(summary,index2show='sigma_beta'):
    mask2 = [index2show in x for x in summary.index]
    return summary[mask2]


def findMAP(model):
    map_estimate = pm.find_MAP(model=model, method="powell")
    return map_estimate





def noncentered_normal_rw(name, *, dims):    
    innov = pm.Normal(f"innov_{name}", 0, 1, dims=dims)
    σ = pm.HalfNormal(f"σ_{name}", 4.1472418499257655) # HALFNORMAL_SCALE * 2.5
    return pm.Deterministic(name, innov.cumsum() * σ, dims=dims)



def CTdataset2(numbers,AG,ct='kidney',genders=['Men','Women']):
    year_list = []
    ag_list = []
    count_list = []
    gender_list = []
    for j,g in enumerate(genders): 
        for ix,y in enumerate(numbers['futureYears']):
            for i,ag in enumerate(AG):
                year_list.append(y)
                ag_list.append(i)
                gender_list.append(g)
                try: 
                    c = numbers[g]['age'][ag]['cancer_types'][ct]['real'][ix]
                    count_list.append(c)
                except:
                    c = numbers[g]['age'][ag]['cancer_types'][ct]['real'][ix-1] # last year
                    count_list.append(c)
    data = pd.DataFrame({'Year':year_list,'AgeGroup':ag_list,'Gender':gender_list,'Count':count_list})
    #data_agg = (data.groupby(["Year", "Gender", "AgeGroup"])["Count"].agg(("size", "sum")))
    return data


def plotPPC(ppc,x_data,varname='ages'):
    mean = ppc[varname].mean(axis=0)
    std = ppc[varname].std(axis=0)
    plt.plot(x_data,mean,label='mean')
    plt.plot(x_data,mean-std,label='lower')
    plt.plot(x_data,mean+std,label='upper')
    plt.legend()
    plt.show()


##############

# your path to documents
save_path = '/home/usix/Documents/cancer/'

# saveDict(numbers,filename=save_path+'numbers_ssm.json')


# numbers['Men']['age']['all']['cancer_types'].keys()

CT = ['Hodgkin_lymphoma', 'Multiple_myeloma', 'Non-Hodgkin_lymphoma', 'all', 'brain', 'breast', 'colon_rectum', 'esophagus', 'kidney', 'liver_bile_duct', 'lung_trachea_bronchi', 'melanoma', 'other_endocrine_glandes', 
    'other_skin_cancer', 'ovarian', 'pancreas', 'prostata', 'stomach', 'testicle', 'thyroid', 'unspecified_localization', 'urinary_tract', 'uterus_cervix_corpus']



#CT = ['ovarian','uterus_cervix_corpus']
#CT = ['testicle','prostata'] 


genders = ['Men','Women']
AG = ['0_4', '5_9','10_14','15_19','20_24','25_29','30_34','35_39','40_44','45_49','50_54','55_59','60_64','65_69','70_74','75_79','80_84','85+'] # ,'all'


numbers = loadDict(save_path+'numbers_ssm.json')
pop_ag = pd.read_csv(save_path+'cohorts/pop_ag.csv',delimiter=',') 
#### population size per year in age groups ####
#pop_ag.to_csv('/home/usix/Documents/cancer/cohorts/pop_ag.csv',index=False,header=True,sep=',')


def prepareData(numbers,genders,AG,pop_ag,ct='kidney'):
    data = {'dfs':[],'genders':genders,'ct':ct,'dfs_test':[],'dfs_train':[],'apc':[],'cancer':[],'apc_test':[]}
    for gender in genders:
        df = CTdataset2(numbers,AG,ct=ct,genders=[gender])
        pop_list = []
        for i in range(len(df)):
            ele = df.iloc[i]
            y = ele['Year']
            ag = ele['AgeGroup']
            g = ele['Gender']
            mask = (pop_ag['Gender'] == g) & (pop_ag['AgeGroup'] == AG[ag])
            c = pop_ag[mask][str(y)].values[0]
            print(y,ag,g,c)
            pop_list.append(c)
        df['pop_size'] = pop_list
        df['incidence'] = df['Count'].values * 100000. /df['pop_size'].values
        cohort_list = []
        for i in range(len(df)):
            ele = df.iloc[i] # 5 years class age groups
            y = ele['Year']
            ag = AG[ele['AgeGroup']]
            if ag == '85+': y_min = 85; y_max = 90; y_mean = (y_max+y_min)/2
            else: y_min, y_max = ag.split('_'); y_min = int(y_min); y_max = int(y_max); y_mean = (y_max+y_min)/2
            cohort_list.append(int(y-y_mean))
        df['cohort'] = cohort_list
        data['dfs'].append(df)
        df_train = df[df.Year != 2022]
        df_train.reset_index(drop=True, inplace=True)
        df_test = df[df.Year != 1970]
        df_test.reset_index(drop=True, inplace=True)
        data['dfs_train'].append(df_train)
        data['dfs_test'].append(df_test)
        apc_df = (df_train.groupby(["Year", "cohort","AgeGroup" ])["incidence"].agg(("size", "sum")))
        data['N'] = apc_df["size"].values
        data['cancer'].append(apc_df["sum"].values)
        i, age_map = apc_df.index.get_level_values("AgeGroup").factorize(sort=True)
        j, year_map = apc_df.index.get_level_values("Year").factorize(sort=True)
        k, cohort_map = apc_df.index.get_level_values("cohort").factorize(sort=True)
        apc_df.index.set_levels(AG, level='AgeGroup',inplace=True) # rename AgeGroup labels 
        age_map = np.array(AG)
        apc_df_all = (df.groupby(["Year", "cohort","AgeGroup" ])["incidence"].agg(("size", "sum")))
        j_all, year_map_all = apc_df_all.index.get_level_values("Year").factorize(sort=True)
        #apc_df_all.index.set_levels(AG, level='AgeGroup',inplace=True)
        apc_df_test = (df_test.groupby(["Year", "cohort","AgeGroup" ])["incidence"].agg(("size", "sum")))
        #cancer_test = apc_df_test["sum"].values
        apc_df_test.index.set_levels(AG, level='AgeGroup',inplace=True)
        data['i'] = i; data['j'] = j ;data['k'] = k; data['j_all'] = j_all
        data['age_map'] = age_map; data['year_map'] = year_map; data['year_map_all'] = year_map_all
        data['cohort_map'] = cohort_map; 
        data['apc'].append(apc_df)
        data['apc_test'].append(apc_df_test)
    return data



def plot3d(yy,data,ct='kidney',gender='Men',bars = False,plotname=''):
    x_ticks = []
    for ee,ele in enumerate(data['year_map_all']):
        if ele%10 ==0: x_ticks.append(str(ele))
        else: x_ticks.append('')
    y = yy.copy()
    plt.close()
    fig = plt.figure(figsize=(18,12))
    axes = fig.add_subplot(projection='3d')
    axes.set_xlabel("Year", fontsize=12)
    axes.set_ylabel("Age Group", fontsize=12)
    axes.set_zlabel("Incidence", fontsize=12)
    axes.set_title('Cancer type:'+ct+' of '+gender,fontsize=14)
    xs = np.arange(y.shape[0])
    if not bars:
        for i,ele in enumerate(data['age_map']):
            axes.plot(xs, np.ones(y.shape[0])*i,y[:,i],label=ele)
    else:
        for j,ag in enumerate(data['age_map']):
            axes.bar(xs, y[:,j], zs=j, zdir='y', alpha=0.7)
    axes.set(yticks=np.arange(len(data['age_map'])), yticklabels=data['age_map'])
    axes.set(xticks=np.arange(len(x_ticks)), xticklabels=x_ticks)
    plt.legend(loc='upper left'); plt.tight_layout()
    if plotname == '': plt.show()
    else: plt.savefig(plotname)




def plotPPC2(ppc_list,data,ct,varname='ages',plotname=''):
    if varname=='ages': x_data = data['age_map']
    if varname=='years': x_data = data['year_map_all']
    if varname=='cohorts': x_data = data['cohort_map']
    colors=['blue','red'] #colors=['cyan','pink']
    alpha=0.5
    plt.close(); plt.figure(figsize = (18, 10))
    for ii,gender in enumerate(data['genders']):
        mean = ppc_list[ii][varname].mean(axis=0)
        std = ppc_list[ii][varname].std(axis=0)
        plt.plot(x_data,mean,label='mean ' +gender,c=colors[ii],alpha=alpha)
        plt.plot(x_data,mean-std,label='lower '+gender,ls="--",c=colors[ii],alpha=alpha)
        plt.plot(x_data,mean+std,label='upper '+gender,ls="--",c=colors[ii],alpha=alpha)
    plt.legend();plt.title('Cancer type:'+ct+' Factor plot of '+varname);plt.xlabel(varname)
    if plotname == '': plt.show()
    else: plt.savefig(plotname)

# CT = [ 'urinary_tract']



ct = 'urinary_tract'
genders = ['Men','Women']



AG = ['0_4', '5_9','10_14','15_19','20_24','25_29','30_34','35_39','40_44','45_49','50_54','55_59','60_64','65_69','70_74','75_79','80_84','85+'] # ,'all'


data = prepareData(numbers,genders,AG,pop_ag,ct=ct)

SAMPLE_KWARGS = {
    "draws": 500,
    "tune": 6000,
    'cores': 4,
    #'random_seed': [SEED + i for i in range(CHAINS)],
    'return_inferencedata': True,
    'target_accept': 0.95
    }




coords = {
    "age": data['age_map'],
    "year": data['year_map'],
    "cohort": data['cohort_map'],
    "year_all": data['year_map_all'],
    "age_groups": len(data['age_map']), # 18
}


years_mask = data['j_all'] != data['j_all'][-1]
years_mask_fut = data['j_all'] != data['j_all'][0]



ppc_list=[];trace_list=[];model_list=[];summary_list=[]

for ii,gender in enumerate(data['genders']):
    with pm.Model(coords=coords) as model:
        N_ = pm.Data("N", data['N'])
        alpha = pm.Normal("alpha", 0, 2.5)
        ages = noncentered_normal_rw("ages", dims="age")
        years = noncentered_normal_rw("years", dims="year_all")
        cohorts = noncentered_normal_rw("cohorts", dims="cohort")
        lamb = pm.Deterministic("lamb",alpha + ages[data['i']] + years[data['j_all'][years_mask]] + cohorts[data['k']]   ) #+ cohorts[data['k']]
        alfa = pm.HalfNormal("alfa", 2.5)
        pm.NegativeBinomial("cancer", N_ * tt.exp(lamb), alfa, observed=data['cancer'][ii]) 
    #start = findMAP(model)
    with model:
        trace = pm.sample(**SAMPLE_KWARGS)
    summary = az.summary(trace)
    summary_list.append(summary)
    trace_list.append(trace)
    with model:
        lamb_fut = pm.Deterministic("lamb_fut",alpha + ages[data['i']] + years[data['j_all'][years_mask_fut]] + cohorts[data['k']]  ) 
        cancer_fut = pm.NegativeBinomial("cancer_fut", N_ * tt.exp(lamb_fut), alfa, shape=len(N)) 
    ppc = pm.sample_posterior_predictive(trace, model=model,var_names=["alpha","ages","years","cancer","cancer_fut","alfa","cohorts"])
    ppc_list.append(ppc)
    model_list.append(model)
    






#out_dict = gradientPlot(ppc_list[1],t_data=[data['age_map'],data['year_map_all'],data['cohort_map']],varnames=['ages','years','cohorts'],comp=['Age','Year','Cohort'])
#graph = pm.model_to_graphviz(model)
#graph.view()



plotPPC2(ppc_list,data,ct=ct,varname='ages',plotname=save_path+'cohorts/'+ct+'_AgeGroups.png')
plotPPC2(ppc_list,data,ct=ct,varname='years',plotname=save_path+'cohorts/'+ct+'_years.png')
plotPPC2(ppc_list,data,ct=ct,varname='cohorts',plotname=save_path+'cohorts/'+ct+'_cohorts.png')



incidence_list = []
count_list = []

for ii,gender in enumerate(data['genders']):
    apc_df = data['apc'][ii].copy()
    apc_df_test = data['apc_test'][ii].copy()
    apc_df["mean"] = ppc_list[ii]['cancer'].mean(axis=0)
    apc_df["lower"] = apc_df["mean"] - ppc_list[ii]['cancer'].std(axis=0)*2.
    apc_df["upper"] = apc_df["mean"] + ppc_list[ii]['cancer'].std(axis=0)*2.
    apc_df_test["mean"] = ppc_list[ii]['cancer_fut'].mean(axis=0)
    apc_df_test["lower"] = apc_df_test["mean"] - ppc_list[ii]['cancer_fut'].std(axis=0)*2.
    apc_df_test["upper"] = apc_df_test["mean"] + ppc_list[ii]['cancer_fut'].std(axis=0)*2.
    first_year = data['year_map_all'][0]
    last_year = data['year_map_all'][-1]
    incidence_dict = {'y_real': [],'y_pred': [],'lower':[],'upper':[]}
    count_dict = {'y_pred': [],'lower':[],'upper':[]}
    for aa,ag in enumerate(data['age_map']):  # shape (year,ageGroup)
        mask = (apc_df_test.index.get_level_values("AgeGroup") == ag)
        mask_df = (data['dfs'][ii]['AgeGroup'] == aa) #& (data['dfs'][ii]['Year'] != 2022)
        mask0 = (apc_df.index.get_level_values("Year") == first_year) & (apc_df.index.get_level_values("AgeGroup") == ag)
        mean = np.concatenate((apc_df[mask0]['mean'].values,apc_df_test[mask]['mean'].values))
        lower = np.concatenate((apc_df[mask0]['lower'].values,apc_df_test[mask]['lower'].values))
        upper = np.concatenate((apc_df[mask0]['upper'].values,apc_df_test[mask]['upper'].values))
        real = np.concatenate((apc_df[mask0]['sum'].values,apc_df_test[mask]['sum'].values))
        pop = data['dfs'][ii][mask_df]['pop_size'].values
        incidence_dict['y_real'].append(real.reshape(-1,1))
        incidence_dict['y_pred'].append(mean.reshape(-1,1))
        incidence_dict['lower'].append(lower.reshape(-1,1))
        incidence_dict['upper'].append(upper.reshape(-1,1))
        mean = mean * pop/100000.
        lower  =  lower * pop/100000.
        upper = upper * pop/100000.
        count_dict['y_pred'].append(mean.reshape(-1,1))
        count_dict['lower'].append(lower.reshape(-1,1))
        count_dict['upper'].append(upper.reshape(-1,1) )
    incidence_dict['y_real'] = np.concatenate(incidence_dict['y_real'],axis=1)
    incidence_dict['y_pred'] = np.concatenate(incidence_dict['y_pred'],axis=1)
    incidence_dict['lower'] = np.concatenate(incidence_dict['lower'],axis=1)
    incidence_dict['upper'] = np.concatenate(incidence_dict['upper'],axis=1)
    count_dict['y_pred'] = np.concatenate(count_dict['y_pred'],axis=1).astype(int)
    count_dict['lower'] = np.concatenate(count_dict['lower'],axis=1).astype(int)
    count_dict['upper'] = np.concatenate(count_dict['upper'],axis=1).astype(int)
    ## incidence can't be negative
    incidence_dict['lower'] = np.where(incidence_dict['lower'] > 0, incidence_dict['lower'], 0)
    count_dict['lower'] = np.where(count_dict['lower'] > 0, count_dict['lower'], 0)
    incidence_list.append(incidence_dict)
    count_list.append(count_dict)
    

for ii,gender in enumerate(data['genders']):
    plot3d(yy=incidence_list[ii]['y_pred'],data=data,ct=ct,gender=gender,bars = False,plotname=save_path+'cohorts/'+ct+'_'+gender+'.png')
    for aa,ag in enumerate(data['age_map']): 
        numbers[gender]['age'][ag]['cancer_types'][ct]['incidence_mean'] = incidence_list[ii]['y_pred'][:,aa].tolist()
        numbers[gender]['age'][ag]['cancer_types'][ct]['incidence_lower'] = incidence_list[ii]['lower'][:,aa].tolist()
        numbers[gender]['age'][ag]['cancer_types'][ct]['incidence_upper'] = incidence_list[ii]['upper'][:,aa].tolist()
        numbers[gender]['age'][ag]['cancer_types'][ct]['count_mean'] = count_list[ii]['y_pred'][:,aa].tolist()
        numbers[gender]['age'][ag]['cancer_types'][ct]['count_lower'] = count_list[ii]['lower'][:,aa].tolist()
        numbers[gender]['age'][ag]['cancer_types'][ct]['count_upper'] = count_list[ii]['upper'][:,aa].tolist()


saveDict(numbers,filename=save_path+'cohorts/numbers_pymc.json')


for ii,gender in enumerate(data['genders']):
    summary_list[ii].to_csv(save_path+'cohorts/'+ct+'_'+gender+'_summary.csv',index=False,header=True,sep=',')




"""
numbers['Men']['age']['70_74']['cancer_types']['prostata'].keys()

for key in ['real', 'ssm_fit2021_predict2022_mean', 'ssm_mean', 'count_mean', ]:
    plt.plot(numbers['Men']['age']['70_74']['cancer_types']['prostata'][key],label=key)

plt.legend()
plt.show()

numbers['Men']['age']['70_74']['cancer_types']['prostata'].keys()
dict_keys(['deviation', 'diff', 'real', 'real_pct', 'real_pct_stand', 'ssm_fit2021_predict2022_deviation', 'ssm_fit2021_predict2022_diff', 'ssm_fit2021_predict2022_lower', 'ssm_fit2021_predict2022_mean', 
'ssm_fit2021_predict2022_upper', 'ssm_lower', 'ssm_mean', 'ssm_upper', 'incidence_mean', 'incidence_lower', 'incidence_upper', 'count_mean', 'count_lower', 'count_upper'])


for key in ['incidence_mean', 'incidence_lower', 'incidence_upper']:
    plt.plot(numbers['Men']['age']['70_74']['cancer_types']['prostata'][key],label=key)

plt.plot(numbers2['Men']['age']['70_74']['cancer_types']['prostata']['real'],label='real')


plt.legend()
plt.show()


numbers2 = loadDict(save_path+'incidence/numbers.json')
"""

numbers['pop_size'] = {'Men':{},'Women':{}}

col =[str(x) for x in range(1970,2022+1)]

for gender in ['Men','Women']:
    all_ag = np.zeros(len(numbers['futureYears']),dtype=int)
    for ag in numbers[gender]['age'].keys():
        mask = (pop_ag['Gender'] == gender) & (pop_ag['AgeGroup'] == ag)
        c = pop_ag[mask][col].values.flatten()
        numbers['pop_size'][gender][ag] = c.tolist()
        if len(c) != 0: all_ag += c
    numbers['pop_size'][gender]['all'] = all_ag.tolist()



men_ct = ['prostata','other_skin_cancer','colon_rectum','urinary_tract','melanoma','lung_trachea_bronchi','Non-Hodgkin_lymphoma','kidney','liver_bile_duct','pancreas','brain', 'stomach', 
   'Hodgkin_lymphoma', 'Multiple_myeloma',  'breast', 'esophagus','other_endocrine_glandes','testicle', 'thyroid', 'unspecified_localization' ]


women_ct = ['breast','other_skin_cancer','uterus_cervix_corpus','colon_rectum','urinary_tract','melanoma','lung_trachea_bronchi','brain','esophagus','kidney', 'liver_bile_duct', 'other_endocrine_glandes', 
     'Hodgkin_lymphoma','Multiple_myeloma', 'Non-Hodgkin_lymphoma','ovarian', 'pancreas', 'stomach', 'thyroid', 'unspecified_localization',]

menwomen_ct = ['breast','other_skin_cancer','colon_rectum','melanoma','lung_trachea_bronchi','urinary_tract','esophagus','kidney','liver_bile_duct','brain','other_endocrine_glandes', 
    'pancreas','stomach','thyroid','Hodgkin_lymphoma','Multiple_myeloma','Non-Hodgkin_lymphoma','unspecified_localization']



for gender in ['Men','Women']:
    for ag in AG: #numbers[gender]['age'].keys():
        if gender == 'Men': gender_ct = men_ct
        if gender == 'Women': gender_ct = women_ct
        if gender == 'Men&Women':gender_ct = menwomen_ct
        for ct in gender_ct : #+ ['all']
            print(gender,ag,ct)
            real = numbers[gender]['age'][ag]['cancer_types'][ct]['real']
            real = np.array(real+[real[-1]]) # add last value of 2021 as value for 2022
            mean = np.array(numbers[gender]['age'][ag]['cancer_types'][ct]['count_mean'])
            lower = np.array(numbers[gender]['age'][ag]['cancer_types'][ct]['count_lower'])
            upper = np.array(numbers[gender]['age'][ag]['cancer_types'][ct]['count_upper'])
            diff = mean-real
            devi = np.round(diff/((0.5*(upper-lower))+0.01),1)
            numbers[gender]['age'][ag]['cancer_types'][ct]['deviation_bayes'] = devi.tolist()
            numbers[gender]['age'][ag]['cancer_types'][ct]['diff_bayes'] = diff.tolist()
            pop = np.array(numbers['pop_size'][gender][ag])
            inci = 100000.*real/pop
            numbers[gender]['age'][ag]['cancer_types'][ct]['incidence_real'] = inci.tolist()
            mean = np.array(numbers[gender]['age'][ag]['cancer_types'][ct]['incidence_mean'])
            lower = np.array(numbers[gender]['age'][ag]['cancer_types'][ct]['incidence_lower'])
            upper = np.array(numbers[gender]['age'][ag]['cancer_types'][ct]['incidence_upper'])
            diff = mean-inci
            devi = np.round(diff/((0.5*(upper-lower))+0.01),1)
            numbers[gender]['age'][ag]['cancer_types'][ct]['deviation_incidence'] = devi.tolist()
            numbers[gender]['age'][ag]['cancer_types'][ct]['diff_incidence'] = diff.tolist()



saveDict(numbers,filename=save_path+'cohorts/numbers_pymc3.json')


########## table output of predictions for 2022
AGshort = ['20_24', '25_29', '30_34', '35_39', '40_44', '45_49', '50_54', '55_59', '60_64', '65_69', '70_74', '75_79', '80_84', '85+']

header = '|Value|CT|Gender|20_24|25_29|30_34|35_39|40_44|45_49|50_54|55_59|60_64|65_69|70_74|75_79|80_84|85+|\n'
header = header + '--'.join(['|']*(len(header.split('|'))-1)) + '\n'

all_ct = ['breast','other_skin_cancer','uterus_cervix_corpus','colon_rectum','urinary_tract','melanoma','lung_trachea_bronchi','brain','esophagus','kidney', 'liver_bile_duct', 'other_endocrine_glandes', 
     'Hodgkin_lymphoma','Multiple_myeloma', 'Non-Hodgkin_lymphoma','ovarian', 'pancreas', 'stomach', 'thyroid', 'unspecified_localization','prostata','testicle']

short_dict = {'other_skin_cancer': 'OSC','uterus_cervix_corpus': 'uterus','colon_rectum':'colon','urinary_tract':'UT','lung_trachea_bronchi':'lung','liver_bile_duct':'liver','other_endocrine_glandes':'OEG',
    'Hodgkin_lymphoma':'HL','Multiple_myeloma':'MM', 'Non-Hodgkin_lymphoma':'NHL','unspecified_localization':'UL'}

val = ['count_lower', 'count_mean', 'count_upper']
val = ['count_lower', 'real','count_mean', 'count_upper']
#val = ['incidence_lower', 'incidence_mean', 'incidence_upper']

#all_ct = ['breast','testicle']




year2show = 2020
ix = numbers['futureYears'].index(year2show)

out_string = ''
for ct in all_ct :
    if (ct in men_ct) & (ct in women_ct): genders = ['Men','Women']
    if (ct in men_ct) & (ct not in women_ct): genders = ['Men']
    if (ct not in men_ct) & (ct in women_ct): genders = ['Women']
    for g in genders:
        for v in val:
            if ct in short_dict.keys(): ct_ = short_dict[ct]
            else: ct_ = ct
            out = '|' + v.split('_')[-1] + '|' + ct_ + '|' + g + '|' 
            for ag in AGshort: 
                out += str(int(numbers[g]['age'][ag]['cancer_types'][ct][v][ix])) + '|'
            out += '\n'
            out_string += out



with open(save_path+'cohorts/incidence2022.txt', 'w') as output:
    output.write(header+out_string)


with open(save_path+'cohorts/count2020.txt', 'w') as output:
    output.write(header+out_string)






def plotBarChart(numbers,col2show=['real','predicted'],year2show=2020,gender='Men',gender_ct=men_ct,ag='all',label='count_'): # label='ssm_fit2021_predict2022_'  label='ssm_'
    ix = numbers['futureYears'].index(year2show) # show results of this year
    predicted_mean = [numbers[gender]['age'][ag]['cancer_types'][ct][label+'mean'][ix] for ct in gender_ct]
    predicted_lower = [numbers[gender]['age'][ag]['cancer_types'][ct][label+'lower'][ix] for ct in gender_ct]
    predicted_upper = [numbers[gender]['age'][ag]['cancer_types'][ct][label+'upper'][ix] for ct in gender_ct]
    try: real = [numbers[gender]['age'][ag]['cancer_types'][ct]['real'][ix] for ct in gender_ct]
    except: real = [numbers[gender]['age'][ag]['cancer_types'][ct]['real'][ix-1] for ct in gender_ct] # last year
    diff = [numbers[gender]['age'][ag]['cancer_types'][ct]['diff'][ix] for ct in gender_ct]
    colors =  ['#adb0ff', '#ffb3ff', '#90d595', '#e48381', '#aafbff', '#f7bb5f', '#eafb50']
    dim = len(col2show)
    w = 0.3
    dimw = w / dim
    x = np.arange(len(real))
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.barh(x , real,dimw,label='real',color=colors[0], left = 0.001)
    ax.barh(x + dimw, predicted_mean,dimw,label='predicted',color=colors[1], left = 0.001)
    ax.set_yticks(x + dimw / 2)
    ax.set_yticklabels(gender_ct)
    for i, (value, name) in enumerate(zip(real, gender_ct)):
        ax.text(value+20, i,'real:'+str(value)+'\npred:'+str(int(predicted_mean[i])),size=8,ha='left',alpha=0.5)
    ax.set_title('Total amount of new cancer '+str(year2show)+' among '+gender+' in Sweden, Age Groups:'+ag+'  Predicted vs Real',size=14)
    ax.text(1, 0.4, str(year2show)+'\n'+gender+'\n Age Group: '+ag+'\nSweden', transform=ax.transAxes, size=46, ha='right')
    plt.legend()
    plt.show()


plotBarChart(numbers,col2show=['real','predicted'],year2show=2020,gender='Men',gender_ct=men_ct,ag='85+',label='count_')





import seaborn as sns

def showHeatMap(numbers,gender='Men',toshow='deviation',year2show=2020,vmin=None,vmax=None,cbar=True,square=True,robust=True,title='Change in Procent from last Year ',logview=False): # 'Deviation from prediction: count ', 'Deviation from prediction: -1:lower conf, 1:upper conf '
    ix = numbers['Year'].index(year2show)
    #AG = ['all','0_4', '5_9','10_14','15_19','20_24','25_29','30_34','35_39','40_44','45_49','50_54','55_59','60_64','65_69','70_74','75_79','80_84', '85+']
    AG = ['20_24','25_29','30_34','35_39','40_44','45_49','50_54','55_59','60_64','65_69','70_74','75_79','80_84', '85+']
    if gender == 'Men': gender_ct = men_ct #+ ['all']
    if gender == 'Women': gender_ct = women_ct #+ ['all']
    if gender == 'Men&Women': gender_ct = menwomen_ct #+ ['all']
    data_list = []
    for ct in gender_ct:
        data_list.append([numbers[gender]['age'][ag]['cancer_types'][ct][toshow][ix] for ag in AG])
    data = np.array(data_list)
    if logview: 
        ax = sns.heatmap(data,cmap="coolwarm",center = 0.0,xticklabels=AG,yticklabels=gender_ct,vmin=vmin,vmax=vmax,cbar=True,square=square, norm=SymLogNorm(linthresh=1))
        ax.set_title(title+str(year2show)+' among '+gender+' in Sweden',size=14)
    else: 
        ax = sns.heatmap(data,cmap="coolwarm",center = 0.0,xticklabels=AG,yticklabels=gender_ct,vmin=vmin,vmax=vmax,cbar=True,square=square,robust=robust)
        ax.set_title(title+str(year2show)+' among '+gender+' in Sweden',size=14)
    #plt.setp(labels, rotation=90)
    plt.xlabel('Age Groups', fontsize = 15) 
    plt.ylabel('Cancer Types', fontsize = 15) 
    plt.show()


showHeatMap(numbers,gender='Women',toshow='diff_incidence',year2show=2020,vmin=None,vmax=None,cbar=True,square=False,robust=True,title='Difference of incidence: blue: real > predicted, red: predicted > real ',logview=False)
showHeatMap(numbers,gender='Women',toshow='diff_incidence',year2show=2021,vmin=None,vmax=None,cbar=True,square=False,robust=True,title='Difference of incidence: blue: real > predicted, red: predicted > real ',logview=False)


showHeatMap(numbers,gender='Men',toshow='diff_incidence',year2show=2020,vmin=None,vmax=None,cbar=True,square=False,robust=True,title='Difference of incidence: blue: real > predicted, red: predicted > real ',logview=False)
showHeatMap(numbers,gender='Men',toshow='diff_incidence',year2show=2021,vmin=None,vmax=None,cbar=True,square=False,robust=True,title='Difference of incidence: blue: real > predicted, red: predicted > real ',logview=False)

showHeatMap(numbers,gender='Women',toshow='incidence_mean',year2show=2021,vmin=None,vmax=None,cbar=True,square=False,robust=True,title='Incidence ',logview=True)
showHeatMap(numbers,gender='Men',toshow='incidence_mean',year2show=2021,vmin=None,vmax=None,cbar=True,square=False,robust=True,title='Incidence ',logview=True)







showHeatMap(numbers,gender='Women',toshow='diff_incidence',year2show=2021,vmin=None,vmax=None,cbar=True,square=False,robust=True,title='Deviation from prediction: blue: real > predicted, red: predicted > real ',logview=False)



showHeatMap(numbers,gender='Men',toshow='diff',year2show=2021,vmin=None,vmax=None,cbar=True,square=False,robust=True,title='Deviation from prediction: blue: real > predicted, red: predicted > real ',logview=False)

showHeatMap(numbers,gender='Men',toshow='diff_incidence',year2show=2021,vmin=None,vmax=None,cbar=True,square=False,robust=True,title='Deviation from prediction: blue: real > predicted, red: predicted > real ',logview=False)

showHeatMap(numbers,gender='Men',toshow='diff_bayes',year2show=2021,vmin=None,vmax=None,cbar=True,square=False,robust=True,title='Deviation from prediction: blue: real > predicted, red: predicted > real ',logview=False)

showHeatMap(numbers,gender='Men',toshow='deviation_incidence',year2show=2021,vmin=None,vmax=None,cbar=True,square=False,robust=True,title='Deviation from prediction: blue: real > predicted, red: predicted > real ',logview=False)
showHeatMap(numbers,gender='Men',toshow='deviation_incidence',year2show=2020,vmin=None,vmax=None,cbar=True,square=False,robust=True,title='Deviation from prediction: blue: real > predicted, red: predicted > real ',logview=False)

showHeatMap(numbers,gender='Women',toshow='deviation_incidence',year2show=2020,vmin=None,vmax=None,cbar=True,square=False,robust=True,title='Deviation from prediction: blue: real > predicted, red: predicted > real ',logview=False)
showHeatMap(numbers,gender='Women',toshow='deviation_incidence',year2show=2021,vmin=None,vmax=None,cbar=True,square=False,robust=True,title='Deviation from prediction: blue: real > predicted, red: predicted > real ',logview=False)

showHeatMap(numbers,gender='Women',toshow='diff_incidence',year2show=2021,vmin=None,vmax=None,cbar=True,square=False,robust=True,title='Deviation from prediction: blue: real > predicted, red: predicted > real ',logview=False)
showHeatMap(numbers,gender='Women',toshow='diff_incidence',year2show=2020,vmin=None,vmax=None,cbar=True,square=False,robust=True,title='Difference pred-real: blue: real > predicted, red: predicted > real ',logview=False)



showHeatMap(numbers,gender='Women',toshow='incidence_mean',year2show=2021,vmin=None,vmax=None,cbar=True,square=False,robust=True,title='Incidence ',logview=True)


showHeatMap(numbers,gender='Men',toshow='deviation_incidence',year2show=2020,vmin=-3,vmax=3,cbar=True,square=False,robust=True,title='Deviation from prediction: blue: real > predicted, red: predicted > real ',logview=False)
showHeatMap(numbers,gender='Men',toshow='deviation_incidence',year2show=2021,vmin=-2,vmax=2,cbar=True,square=False,robust=True,title='Deviation from prediction: blue: real > predicted, red: predicted > real ',logview=False)


showHeatMap(numbers,gender='Men',toshow='diff',year2show=2020,vmin=None,vmax=None,cbar=True,square=False,robust=True,title='Difference (count) pred-real: blue: real > predicted, red: predicted > real ',logview=False)
showHeatMap(numbers,gender='Men',toshow='diff',year2show=2021,vmin=None,vmax=None,cbar=True,square=False,robust=True,title='Difference (count) pred-real: blue: real > predicted, red: predicted > real ',logview=False)

showHeatMap(numbers,gender='Women',toshow='diff',year2show=2020,vmin=None,vmax=None,cbar=True,square=False,robust=True,title='Difference (count) pred-real: blue: real > predicted, red: predicted > real ',logview=False)
showHeatMap(numbers,gender='Women',toshow='diff',year2show=2021,vmin=None,vmax=None,cbar=True,square=False,robust=True,title='Difference (count) pred-real: blue: real > predicted, red: predicted > real ',logview=False)





def showCancerType(numbers,AGr,gender='Men',ct='prostata',val='real',incidence=False):
    #AGr = [('all'),('0_4', '5_9','10_14','15_19','20_24'),('25_29','30_34','35_39','40_44'),('45_49','50_54','55_59','60_64','65_69'),('70_74','75_79','80_84','85+')]
    #AGr = [('all'),('25_29','30_34','35_39','40_44'),('45_49','50_54','55_59','60_64','65_69'),('70_74','75_79','80_84','85+')]
    #AGr = [('25_29','30_34','35_39','40_44'),('45_49','50_54','55_59','60_64','65_69'),('70_74','75_79','80_84','85+')]
    num = len(AGr)
    if val !='real': dates = [dt.date(year=x,month=1,day=1) for x in numbers['futureYears']]
    else: dates = [dt.date(year=x,month=1,day=1) for x in numbers['Year']]
    fig, axes = plt.subplots(num,1,sharex=True,figsize=(16,10))
    fig.autofmt_xdate()
    for i,a in enumerate(AGr):
        if incidence: axes[i].set_title('Incidence of ' +ct+' among '+gender+' Age groups:'+str(a)+' shows '+val+' values')
        else: axes[i].set_title('Total count of ' +ct+' among '+gender+' Age groups:'+str(a)+' shows '+val+' values')
        if a == 'all': axes[i].plot(dates, numbers[gender]['age'][a]['cancer_types'][ct][val],label=a)
        else:
            for j,ag in enumerate(a):
                axes[i].plot(dates, numbers[gender]['age'][ag]['cancer_types'][ct][val],label=ag)
        xfmt = mdates.DateFormatter('%Y')
        axes[i].xaxis.set_major_formatter(xfmt)
        axes[i].legend(fontsize=8)
    plt.tight_layout()
    plt.show()

AGr = [('25_29','30_34','35_39','40_44'),('45_49','50_54','55_59','60_64','65_69'),('70_74','75_79','80_84','85+')]
showCancerType(numbers,AGr=AGr,gender='Women',ct='breast',val='incidence_mean',incidence=True)


showCancerType(numbers,AGr=AGr,gender='Men',ct='prostata',val='incidence_mean',incidence=True)

showCancerType(numbers,AGr=AGr,gender='Men',ct='stomach',val='incidence_mean',incidence=True)

showCancerType(numbers,AGr=AGr,gender='Men',ct='prostata',val='diff_incidence',incidence=True)




AGr = [('all'),('25_29','30_34','35_39','40_44'),('45_49','50_54','55_59','60_64','65_69'),('70_74','75_79','80_84','85+')]
showCancerType(numbers,AGr=AGr,gender='Men',ct='prostata',val='real',incidence=False)

showCancerType(numbers,AGr=AGr,gender='Women',ct='breast',val='real',incidence=False)

showCancerType(numbers,AGr=AGr,gender='Women',ct='breast',val='diff',incidence=False)




def plotTable(numbers,gender='Men',year2show=2020,showCT = True,ag='all',ct='prostata',rows = ['real','ssm_mean','ssm_lower','ssm_upper','diff'],maxcol=9,figsize=(18,2)):
    #rows = ['real','fitted_2019_predicted_mean','fitted_2019_predicted_lower','fitted_2019_predicted_upper','fitted_2019_diff']
    ix = numbers['Year'].index(year2show)
    AG = ['0_4', '5_9','10_14','15_19','20_24','25_29','30_34','35_39','40_44','45_49','50_54','55_59','60_64','65_69','70_74','75_79','80_84','85+']
    AG = AG[-maxcol:]
    if gender == 'Men': gender_ct = men_ct # gender_ct = ['all'] + men_ct
    if gender == 'Women': gender_ct = women_ct
    if gender == 'Men&Women': gender_ct =  menwomen_ct
    gender_ct =gender_ct[:maxcol] 
    if showCT: columns = gender_ct
    else: columns = AG
    data = []
    cell_text = []
    for r in rows:
        if showCT: 
            d = [numbers[gender]['age'][ag]['cancer_types'][c][r][ix] for c in gender_ct]
            data.append(d)
            #cell_text.append(['%1.1f' % (x ) for x in d])
            cell_text.append([str(int(x)) for x in d])
        else: 
            d = [numbers[gender]['age'][a]['cancer_types'][ct][r][ix] for a in AG]
            data.append(d)
            cell_text.append([str(int(x)) for x in d])
    colors = plt.cm.BuPu(np.linspace(0, 0.5, len(rows)))
    colors = colors[::-1]
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_visible(False)
    #plt.gca().axis('off')
    ax.axis('off')
    ax.axis('tight')
    the_table = ax.table(cellText=cell_text,rowLabels=rows,rowColours=colors,colLabels=columns,loc='center',cellLoc='center') # ,fontsize=12
    if showCT: plt.title('Total count of Cancer Type ' +' among '+gender+' Age groups:'+str(ag)+' in '+str(year2show))
    else: plt.title('Total count of Cancer Type ' +ct+' among '+gender+' in '+str(year2show))
    fig.tight_layout()
    plt.show()


plotTable(numbers,gender='Men',year2show=2020,showCT=True,ag='all',ct='prostata',rows = ['real','ssm_mean','ssm_lower','ssm_upper','diff'],maxcol=9,figsize=(18,2))

plotTable(numbers=numbers2,gender='Men',year2show=2020,showCT=True,ag='85+',ct='prostata',rows = ['real','count_lower', 'count_mean', 'count_upper'],maxcol=9,figsize=(18,2))

plotTable(numbers=numbers2,gender='Men',year2show=2020,showCT=False,ag='85+',ct='prostata',rows = ['real','count_lower', 'count_mean', 'count_upper'],maxcol=9,figsize=(18,2))


#numbers2 = loadDict(filename=save_path+'cohorts/numbers_pymc3.json')



numbers2['Men']['age']['85+']['cancer_types']['prostata'].keys()

















#############################################
ii = 0
gender = data['genders'][ii]

# for ii,gender in enumerate(data['genders']):
with pm.Model(coords=coords) as model:
    N_ = pm.Data("N", data['N'])
    alpha = pm.Normal("alpha", 0, 2.5)
    ages = noncentered_normal_rw("ages", dims="age")
    years = noncentered_normal_rw("years", dims="year_all")
    cohorts = noncentered_normal_rw("cohorts", dims="cohort")
    lamb = pm.Deterministic("lamb",alpha + ages[data['i']] + years[data['j_all'][years_mask]] + cohorts[data['k']]   ) #+ cohorts[data['k']]
    alfa = pm.HalfNormal("alfa", 2.5)
    pm.NegativeBinomial("cancer", N_ * tt.exp(lamb), alfa, observed=data['cancer'][ii]) 




start = findMAP(model)




with model:
    trace = pm.sample(**SAMPLE_KWARGS)


showRhat(trace)
#az.plot_trace(trace);plt.show()
az.loo(trace,var_name='cancer')



with model:
    lamb_fut = pm.Deterministic("lamb_fut",alpha + ages[data['i']] + years[data['j_all'][years_mask_fut]] + cohorts[data['k']]  ) 
    cancer_fut = pm.NegativeBinomial("cancer_fut", N_ * tt.exp(lamb_fut), alfa, shape=len(N)) 


ppc = pm.sample_posterior_predictive(trace, model=model,var_names=["alpha","ages","years","cancer","cancer_fut","alfa","cohorts"])


plotPPC(ppc,x_data=data['age_map'],varname='ages')
plotPPC(ppc,x_data=data['year_map_all'],varname='years')
plotPPC(ppc,x_data=data['cohort_map'],varname='cohorts')

saveResultsAndPlot(ppc,df=data['apc'][ii],df_test=data['apc_test'][ii],AG=AG,CI_WIDTH = 0.95,show=True,ag='70_74',factor=1.,fit_label='cancer',fut_label='cancer_fut')

out_dict = gradientPlot(ppc,t_data=[data['age_map'],data['year_map_all'],data['cohort_map']],varnames=['ages','years','cohorts'],comp=['Age','Year','Cohort'])




#############################################################################
with pm.Model(coords=coords) as model:
    N_ = pm.Data("N", data['N'])
    alphas =[];ages=[];years=[];cohorts=[];lambs=[];alfas=[];likes=[]
    for ii,gender in enumerate(data['genders']):
        alphas.append(pm.Normal("alpha_"+gender, 0, 2.5))
        ages.append(noncentered_normal_rw("ages_"+gender, dims="age"))
        years.append(noncentered_normal_rw("years_"+gender, dims="year_all"))
        cohorts.append(noncentered_normal_rw("cohort_"+gender, dims="cohort"))
        lambs.append(pm.Deterministic("lamb_"+gender,alphas[ii] + ages[ii][data['i']] + years[ii][data['j_all'][years_mask]]  + cohorts[ii][data['k']]  ) )
        alfas.append(pm.HalfNormal("alfa_"+gender, 2.5))
        likes.append(pm.NegativeBinomial("cancer_"+gender, N_ * tt.exp(lambs[ii]), alfas[ii], observed=data['cancer'][ii]) )


start = findMAP(model)




with model:
    trace = pm.sample(**SAMPLE_KWARGS)


showRhat(trace)
#az.plot_trace(trace);plt.show()
az.loo(trace,var_name='cancer_Men')
az.loo(trace,var_name='cancer_Women')

with model:
    lambs_fut = []; likes_fut = []
    for ii,gender in enumerate(data['genders']):
        lambs_fut.append( pm.Deterministic("lamb_fut_"+gender,alphas[ii] + ages[ii][data['i']] + years[ii][data['j_all'][years_mask_fut]]  + cohorts[ii][data['k']]  ) )
        likes_fut.append( pm.Normal("cancer_fut_"+gender, mu=lambs_fut[ii], sigma=alfas[ii], shape=len(N)) )


ppc = pm.sample_posterior_predictive(trace, model=model,var_names=["alpha_Men","alpha_Women","ages_Men","ages_Women","years_Men","years_Women","cohort_Men","cohort_Women","cancer_Men","cancer_Women","cancer_fut_Men","cancer_fut_Women"])


plotPPC(ppc,x_data=age_map,varname='ages_Men')
plotPPC(ppc,x_data=cohort_map,varname='cohort_Men')
plotPPC(ppc,x_data=year_map_all,varname='years_Men')


quot = (ppc["years_men"]-ppc["years_women"])/(ppc["years_men"]+ppc["years_women"])
plt.plot(year_map_all,quot.mean(axis=0))
plt.show()

quot = (ppc["cohort_men"]-ppc["cohort_women"])/(ppc["cohort_men"]+ppc["cohort_women"])
plt.plot(cohort_map,quot.mean(axis=0))
plt.show()


quot = (ppc["years_men"]/ppc["years_women"])
plt.plot(year_map_all,quot.mean(axis=0))
plt.show()




def saveResultsAndPlot(ppc,df,df_test,AG,CI_WIDTH = 0.95,show=True,ag='70_74',factor=1.,fit_label='cancer',fut_label='cancer_fut'):
    if factor != 1.:
        ppc[fit_label] = ppc[fit_label] * factor
        ppc[fut_label] = ppc[fut_label] * factor
    apc_df = df.copy()
    apc_df_test = df_test.copy()
    apc_df["mean"] = ppc[fit_label].mean(axis=0)
    #apc_df["upper"] = np.percentile(ppc['cancer'],CI_WIDTH,axis=0,)
    #apc_df["lower"] = np.percentile(ppc['cancer'],1.-CI_WIDTH,axis=0,)
    apc_df["upper"] = apc_df["mean"].values + ppc[fit_label].std(axis=0)
    apc_df["lower"] = apc_df["mean"].values - ppc[fit_label].std(axis=0)
    
    if show:
        mask = (apc_df.index.get_level_values("AgeGroup") == ag)
        apc_df[mask]['sum'].plot(label='real');
        apc_df[mask]['mean'].plot(label='predicted')
        apc_df[mask]['upper'].plot(label='upper')
        apc_df[mask]['lower'].plot(label='lower')
        plt.legend()
        plt.show()
    apc_df_test["mean"] = ppc[fut_label].mean(axis=0)
    #apc_df_test["upper"] = np.percentile(ppc['cancer_fut'],CI_WIDTH,axis=0,)
    #apc_df_test["lower"] = np.percentile(ppc['cancer_fut'],1.-CI_WIDTH,axis=0,)
    apc_df_test["upper"] = apc_df_test["mean"].values + ppc[fut_label].std(axis=0)
    apc_df_test["lower"] = apc_df_test["mean"].values - ppc[fut_label].std(axis=0)
    if show:
        mask = (apc_df_test.index.get_level_values("AgeGroup") == ag)
        apc_df_test[mask]['sum'].plot(label='real');
        apc_df_test[mask]['mean'].plot(label='predicted')
        apc_df_test[mask]['upper'].plot(label='upper')
        apc_df_test[mask]['lower'].plot(label='lower')
        plt.legend()
        plt.show()
    return apc_df,apc_df_test

saveResultsAndPlot(ppc,df=apc_df_women,df_test=apc_df_test_women,AG=AG,CI_WIDTH = 0.95,show=True,ag='70_74',factor=1.,fit_label='cancer_women',fut_label='cancer_fut_women')

mask = (apc_df_women.index.get_level_values("AgeGroup") == ag)
apc_df_women[mask]['sum'].plot(label='real');
apc_df = apc_df_women.copy()
apc_df["mean"] = ppc['cancer_women'].mean(axis=0)
apc_df[mask]['mean'].plot(label='predicted')
plt.legend()
plt.show()


apc_df = apc_df_test_women.copy()
mask = (apc_df.index.get_level_values("AgeGroup") == ag)
apc_df[mask]['sum'].plot(label='real');
apc_df["mean"] = ppc['cancer_fut_women'].mean(axis=0)
apc_df[mask]['mean'].plot(label='predicted')
plt.legend()
plt.show()




saveResultsAndPlot(ppc,df=data['apc'][1],df_test=data['apc_test'][1],AG=AG,CI_WIDTH = 0.95,show=True,ag='70_74',factor=1.,fit_label='cancer_Women',fut_label='cancer_fut_Women')





