import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from collections import defaultdict
from collections import Counter
from Bio.SeqUtils.ProtParam import ProteinAnalysis

def file_name(file_dir):
    for root,dirs,files in os.walk(file_dir):
        LL1=[]
        for ff in files:
            if os.path.splitext(ff)[1]=='.csv':
                LL1.append(os.path.join(ff))
        return LL1
    
def get_max_value(martix):
    max_list=[]
    for j in range(len(martix[0])):
        one_list=[]
        for i in range(len(martix)):
            one_list.append(float(martix[i][j]))
        max_list.append(max(one_list))
    return max_list 
def get_min_value(martix):
    max_list=[]
    for j in range(len(martix[0])):
        one_list=[]
        for i in range(len(martix)):
            one_list.append(float(martix[i][j]))
        max_list.append(min(one_list))
    return max_list                  
def MaxMinNormalization(x,Max,Min):
    x = (float(x) - float(Min)) / (float(Max) - float(Min))
    return x


# Encode the AAC feature with the protein sequence
def AAC_feature(fastas):
    AA = 'ACDEFGHIKLMNPQRSTVWY*'
    #AA = 'ARNDCQEGHILKMFPSTWYV'
    encodings = []
    header = ['#']
    for i in AA:
        header.append(i)
    encodings.append(header)
    sequence = fastas
    count = Counter(sequence)
    for key in count:
        count[key] = float(count[key])/len(sequence)
    code = []
    for aa in AA:
        code.append(count[aa])
    encodings.append(code)
    return code

# Extract the physical-chemical properties from the protein sequnce
def physical_chemical_feature(sequence):
    seq_new=sequence.replace('X','').replace('U','').replace('B','').replace('Z','')
    CE = 'CHONS'
    Chemi_stats = {'A':{'C': 3, 'H': 7, 'O': 2, 'N': 1, 'S': 0},
                   'C':{'C': 3, 'H': 7, 'O': 2, 'N': 1, 'S': 1},
                   'D':{'C': 4, 'H': 7, 'O': 4, 'N': 1, 'S': 0},
                   'E':{'C': 5, 'H': 9, 'O': 4, 'N': 1, 'S': 0},
                   'F':{'C': 9, 'H': 11,'O': 2, 'N': 1, 'S': 0},
                   'G':{'C': 2, 'H': 5, 'O': 2, 'N': 1, 'S': 0},
                   'H':{'C': 6, 'H': 9, 'O': 2, 'N': 3, 'S': 0},
                   'I':{'C': 6, 'H': 13,'O': 2, 'N': 1, 'S': 0},
                   'K':{'C': 6, 'H': 14,'O': 2, 'N': 2, 'S': 0},
                   'L':{'C': 6, 'H': 13,'O': 2, 'N': 1, 'S': 0},
                   'M':{'C': 5, 'H': 11,'O': 2, 'N': 1, 'S': 1},
                   'N':{'C': 4, 'H': 8, 'O': 3, 'N': 2, 'S': 0},
                   'P':{'C': 5, 'H': 9, 'O': 2, 'N': 1, 'S': 0},
                   'Q':{'C': 5, 'H': 10,'O': 3, 'N': 2, 'S': 0},
                   'R':{'C': 6, 'H': 14,'O': 2, 'N': 4, 'S': 0},
                   'S':{'C': 3, 'H': 7, 'O': 3, 'N': 1, 'S': 0},
                   'T':{'C': 4, 'H': 9, 'O': 3, 'N': 1, 'S': 0},
                   'V':{'C': 5, 'H': 11,'O': 2, 'N': 1, 'S': 0},
                   'W':{'C': 11,'H': 12,'O': 2, 'N': 2, 'S': 0},
                   'Y':{'C': 9, 'H': 11,'O': 3, 'N': 1, 'S': 0}
                }
    
    count = Counter(seq_new)
    code = []
    
    for c in CE:
        abundance_c = 0
        for key in count:
            num_c = Chemi_stats[key][c]
            abundance_c += num_c * count[key]
            
        code.append(abundance_c)
    return(code)

# calculate the protein molecular for the protein sequnce.
def molecular_weight(seq):
    seq_new=seq.replace('X','').replace('U','').replace('B','').replace('Z','')
    analysed_seq = ProteinAnalysis(seq_new)
    analysed_seq.monoisotopic = True
    mw = analysed_seq.molecular_weight()
    return([mw])
                   
if __name__ == '__main__':
    a=[]
    with open('../data/test-test-seq.fasta','r') as f:
        for i in f:
            a.append(i.split('\n')[0])
    with open('../data/mediumdata/testseq_CHONS.csv','w') as chons,\
    open('../data/mediumdata/testseq_weight.csv','w') as weight,\
    open('../data/mediumdata/testseq_AAC.csv','w') as aac:
        chons.write('#\tC\tH\tO\tN\tS\n')
        weight.write('#\tweight\n')
        aac.write('#\tA\tC\tD\tE\tF\tG\tH\tI\tK\tL\tM\tN\tP\tQ\tR\tS\tT\tV\tW\tY\t*\n')
        for i in range(0,len(a),2):
            chons.write(a[i].strip('>')+'\t'+'\t'.join([str(kk) for kk in physical_chemical_feature(a[i+1])])+'\n')
            weight.write(a[i].strip('>')+'\t'+str(molecular_weight(a[i+1])[0])+'\n')
            aac.write(a[i].strip('>')+'\t'+'\t'.join([str(kk) for kk in AAC_feature(a[i+1])])+'\n')
            
    data=pd.read_csv('../data/test-test.csv',sep='\t')
    pp=[]
    hh=[]
    for i in data.index:
        pp.append(data.ix[i,'phage'])
        hh.append(data.ix[i,'host'])
    pp=list(set(pp))
    hh=list(set(hh))
    features=['CHONS','weight','AAC']
    dic_all={}
    ff=features[0]
    print(ff)
    locals()['dic_'+ff]=defaultdict(list)
    with open('../data/mediumdata/testseq_'+ff+'.csv','r') as f:
        aa=f.readline()
        print(len(aa.split('\t'))-1)
        for line in f:
            locals()['dic_'+ff]['_'.join(line.strip().split('\t')[0].split('_')[:-1])].append([float(mm) for mm in line.strip().split('\t')[1:]])
    locals()['dic2_'+ff]={}
    for x in locals()['dic_'+ff].keys():
        xx=np.array(locals()['dic_'+ff][x])
        dic_all[x]=[xx.mean(axis=0).tolist(),xx.max(axis=0).tolist(),xx.min(axis=0).tolist(),
                 xx.std(axis=0).tolist(),xx.var(axis=0).tolist(),np.median(xx, axis=0).tolist()]
    
    for ff in features[1:]:
    	print(ff)
        locals()['dic_'+ff]=defaultdict(list)
        with open('../data/mediumdata/testseq_'+ff+'.csv','r') as f:
            aa=f.readline()
            print(len(aa.split('\t'))-1)
            for line in f:
                locals()['dic_'+ff]['_'.join(line.strip().split('\t')[0].split('_')[:-1])].append([float(mm) for mm in line.strip().split('\t')[1:]])
        locals()['dic2_'+ff]={}
        for x in locals()['dic_'+ff].keys():
            xx=np.array(locals()['dic_'+ff][x])
            dic_all[x]=[dic_all[x][0]+xx.mean(axis=0).tolist(),dic_all[x][1]+xx.max(axis=0).tolist(),dic_all[x][2]+xx.min(axis=0).tolist(),
                     dic_all[x][3]+xx.std(axis=0).tolist(),dic_all[x][4]+xx.var(axis=0).tolist(),dic_all[x][5]+np.median(xx, axis=0).tolist()] 
    for ii in dic_all.keys():
        np.savetxt('../data/mediumdata/'+ii+'.csv',np.array(dic_all[ii]))
    max_num=pd.read_csv('../data/max_num.csv',header=None).get_values()
    min_num=pd.read_csv('../data/min_num.csv',header=None).get_values()

    data=pd.read_csv('../data/test-test.csv',sep='\t')
    for ii in data.index:
        phage=pd.read_csv('../data/mediumdata/'+data.ix[ii,'phage']+'.csv',header=None,sep=' ').get_values().flatten().tolist()
        host=pd.read_csv('../data/mediumdata/'+data.ix[ii,'host']+'.csv',header=None,sep=' ').get_values().flatten().tolist()
        kk=np.array(phage+host)
        with open('../data/testfeatures/'+data.ix[ii,'phage']+','+data.ix[ii,'host']+'.csv','w') as fout:
            for ii in range(len(kk)):
                if max_num[ii][0]!=min_num[ii][0]:
                    kk[ii]=MaxMinNormalization(kk[ii],max_num[ii][0],min_num[ii][0])
                    fout.write(str(kk[ii])+'\t')
                else:
                    fout.write('0\t')
                fout.write('\n')


          
 























           