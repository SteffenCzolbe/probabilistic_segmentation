from collections import defaultdict
import pandas as pd
import pickle
import numpy as np
import pingouin as pg


def load_data(test_results_file):
    with open(test_results_file, 'rb') as f:
        test_results = pickle.load(f)

    data_samples = defaultdict(list)
    data_class = defaultdict(list)
    data_images = defaultdict(list)
    
    for dataset in test_results.keys():
        for model in test_results[dataset].keys():
            model_uncert = test_results[dataset][model]['per_sample']["test/model_uncertainty"]
            annot_uncert = test_results[dataset][model]['per_sample']["test/annotator_uncertainty"]
            
            #sample data
            for img_id in range(len(model_uncert)):
                uncert = np.array(annot_uncert[img_id])
                N = uncert.shape[0]
                agreement=np.zeros(N)
                agreement[(uncert > 0.) & (uncert < 1.)] = 1
                agreement[uncert >= 1.] = 2
                if np.max(agreement) < 1: continue

                data_samples['model_uncertainty'] += model_uncert[img_id]
                data_samples['annot_uncertainty'] += annot_uncert[img_id]
                data_samples['annot_agreement'] += list(agreement)
                data_samples['image'] += N * [img_id]
                data_samples['model'] += N * [model]
                data_samples['dataset'] += N * [dataset]
                
            #classification data
            tp_uncert = test_results[dataset][model]['per_sample']["test/tp_uncertainty"]
            tn_uncert = test_results[dataset][model]['per_sample']["test/tn_uncertainty"]
            fp_uncert = test_results[dataset][model]['per_sample']["test/fp_uncertainty"]
            fn_uncert = test_results[dataset][model]['per_sample']["test/fn_uncertainty"]
                
            for img_id in range(len(model_uncert)):
                Nt = len(tp_uncert[img_id])+len(tn_uncert[img_id])
                Nf = len(fp_uncert[img_id])+len(fn_uncert[img_id])
                N = Nt + Nf
                if Nt == 0 or Nf == 0: continue

                data_class['model_uncertainty'] += tp_uncert[img_id]
                data_class['model_uncertainty'] += tn_uncert[img_id]
                data_class['model_uncertainty'] += fp_uncert[img_id]
                data_class['model_uncertainty'] += fn_uncert[img_id]
                data_class['correct'] += Nt * [1]
                data_class['correct'] += Nf * [0]
                
                data_class['image'] += N * [img_id]
                data_class['model'] += N * [model]
                data_class['dataset'] += N * [dataset]
            
            #image data
            
            geds=test_results[dataset][model]['per_sample']['test/ged/16']
            N=len(geds)
            data_images['ged'] += list(geds)
            data_images['model'] += N * [model]
            data_images['dataset'] += N * [dataset]
            
    return pd.DataFrame(data_samples), pd.DataFrame(data_class), pd.DataFrame(data_images)
if __name__ == "__main__":
    
    # load data  into pandas dataframes
    data_samples, data_class, data_images = load_data('./plots/experiment_results.pickl')
    # with open('frames.pickl', 'wb') as f:
    #     pickle.dump((data_samples, data_class, data_images),f)
    #with open('frames.pickl', 'rb') as f:
    #    data_samples, data_class, data_images = pickle.load(f)

    print('\ncorrelation model uncertainty with error')
    for dataset in data_class['dataset'].unique():
        data_set = data_class[data_class['dataset']==dataset]
        for model in data_set['model'].unique():
            model_data = data_set[data_set['model']==model]
            corr_res = pg.rm_corr(model_data, x='correct', y='model_uncertainty', subject='image')
        
            print(dataset+"/"+model, float(corr_res['r']), float(corr_res['pval']))
    
    print('\nged, p-values, each pair ged(model1) < ged(model2)')
    for dataset in data_images['dataset'].unique():
        data_set = data_images[data_images['dataset']==dataset]
        for modeli in data_set['model'].unique():
            model_datai = data_set[data_set['model']==modeli]
            for modelj in data_set['model'].unique():
                if modeli == modelj: continue
                model_dataj = data_set[data_set['model']==modelj]
                ged_res = pg.ttest(model_datai['ged'],model_dataj['ged'], tail='less')
                print(dataset+"/"+modeli+"/"+modelj, float(ged_res['p-val']))
    
    print('\ncorrelation uncertainty, agreement')
    for dataset in data_samples['dataset'].unique():
        data_set = data_samples[data_samples['dataset']==dataset]
        for model in data_set['model'].unique():
            model_data = data_set[data_set['model']==model]
            corr_res = pg.rm_corr(model_data, x='annot_agreement', y='model_uncertainty', subject='image')
        
            print(dataset+"/"+model, float(corr_res['r']), float(corr_res['pval']))