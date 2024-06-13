#DATASET LOADING
import evaluate,torch,os
from pathlib import Path

import numpy as np
from datasets import load_dataset,Dataset,DatasetDict
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer,DataCollatorWithPadding
from collections import Counter
from sklearn.metrics import multilabel_confusion_matrix
from torch.utils.data import DataLoader


from nyth_dataset import NYTHDataset
import json
# Create the object with the data path
label2id = None
with open('./data/nyt-h/rel2id.json') as f:
    label2id = json.load(f)
id2label= {}
for i in  label2id:
    id2label[label2id[i]] = i  
dataset_nyth = NYTHDataset(data_dir='./data/nyt-h', include_na_relation=False)
# Load the data
dataset_nyth.load_data(reload=True)
# Get the data
train,dev,test = dataset_nyth.get_data()
mymap = {'relation':'label'}
train = train.rename(columns=mymap)
dev = dev.rename(columns=mymap)
test  = test.rename(columns=mymap)
def prepare_dataset(df):
    df= df.rename(columns=mymap)
    df = df[:100]
    df = Dataset.from_pandas(df)
    return df  

train = Dataset.from_pandas(train)
dev = Dataset.from_pandas(dev)
test = Dataset.from_pandas(test)
#train = prepare_dataset(train)
#dev = prepare_dataset(train)
#test = prepare_dataset(train)

dataset =  DatasetDict({
    'train':train,
    'validation':dev,
    'test':test

})

#ENCODING
# INPUTS = sentence,e1_name,e2_name
# OUTPUTS = relation
#print(train)
#print(train.columns)
#print(train['relation'])
#print(train['bag_label'])
print(type(dataset))
print(train[0])
#train['relation'].rename('label')
#assert(False)
#print(train)
#example = train[0]
labels = list(set(train[f"label"]))

def flatten(l):
    return [item for sublist in l for item in sublist]

metrics_macro_weighted = ['precision','recall','f1'] #macro and spec make no sense in macro/weighted 
avg_types= ['macro','weighted','micro']# just to be consistent
by_class_metrics = ['imbalance','precision','recall','f1','bias_f1','specificity'] 
notations = ['','_bi']#we will calculate stats with and without O of BIO notation
notations = ['']
def write_metrics(m_values):
    out = ""
    tab= '\t'
    x = lambda  x : id2label[x] 
    for n in notations:
        out += f'Classes:\t{tab.join(map(x,m_values[f"labels{n}"]))}\n'
        for m in by_class_metrics:
            out += f'{m.upper()}{n} by class score:\t{tab.join(map(str,m_values[f"{m}_by_class{n}"]))}\n'
        for m in metrics_macro_weighted:
            for avg in avg_types:
                out += f'{m.upper()}_{avg.upper()}{n} score:\t{m_values[f"{m}_{avg}{n}"]}\n'
        for m in avg_types:
            out += f'Bias F1{m.upper()}{n} :\t{m_values[f"bias_f1{m}{n}"]}\n'
                
    model.train()
    with open(datafile+f'/metrics_per_classes_training.csv','a') as f:
        f.write(out)
    #cur_epoch += 1

def compute_metrics(p):
    print(p)
    predictions, labels = p
    #assert(False)
    predictions = np.argmax(predictions, axis=1)

    # Remove ignored index (special tokens)
    y_hat = [
        [p for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    y = [
        [l for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    def flatten(l):
        return [item for sublist in l for item in sublist]
    
    scores = {}
    null_class = 0
    #We will be calculating all metrics and bias
    #in the BIO notation and excluding O(BI notation)
    y = flatten(y)
    y_hat = flatten(y_hat)
    for n in  notations:
        #We set overall vars
        y_classes = list(set(y))
        y_new = y
        y_hat_new = y_hat

        y_classes.remove(null_class)
        num_classes = len(y_classes)
        scores[f'labels'] = y_classes
        
        #Confusion matrix
        cfm = multilabel_confusion_matrix(y_pred=y_hat_new, y_true=y_new, labels = scores['labels'])
        #   0   1   yhat/y  ###
        #   TN  FP  0       ### HOW CONFUSION MATRIX LOOKS LIKE
        #   FN  TP  1       ###
        #Middleware metrics for F1,Bias F1



        for m in by_class_metrics:
            scores[f'{m}_by_class'] = []
        for cl in cfm:
            scores[f'recall_by_class'].append(cl[1][1]/(cl[1][1] +cl[1][0]))
            scores[f'precision_by_class'].append(cl[1][1]/(cl[1][1] +cl[0][1]))
            try:
                recall = scores[f'recall_by_class'][-1]
                precision = scores[f'precision_by_class'][-1]
                scores[f'f1_by_class'].append(2.0/(1.0/recall+1.0/precision))
            except ZeroDivisionError:
                scores[f'f1_by_class'].append(0.0)
            scores[f'specificity_by_class'].append(cl[0][0]/(cl[0][0] +cl[0][1]))
            #since null_class is removed from the confusion matrix alltogether;
            #I think using generally the metrics from 
            scores[f'imbalance_by_class'].append((2*(cl[1][0] +cl[1][1] )/sum(flatten(cl))) - 1)

            #print(cl)
            #print(scores['imbalance_by_class'][-1])
        #assert(False)
        def add_aggregated_metric(m_name,p,r,avg):
            metric = evaluate.load(f'evaluate/metrics/{m_name}/{m_name}.py')
            scores[f'{m_name}_{avg}'] = metric.compute(predictions=p, references=r, average=avg,labels=scores['labels'])[m_name]
            #only needed for best model 
            if m_name == 'f1' and avg == 'macro':
                scores[f'{m_name}'] = metric.compute(predictions=p, references=r, average=avg,labels=scores['labels'])[m_name]

        for avg in  avg_types: 
            for i in metrics_macro_weighted:
                add_aggregated_metric(i,y_hat_new,y_new,avg)
        
        ###BIAS CALCULATION
        # For microF1 we need to accumulate fraction components separately
        microf1_term_names = ['num','denom_unbal','denom_bal']
        for t in microf1_term_names:
            scores[f'microf1_terms_{t}'] = 0.
        #print(num_classes)
        #print(scores['labels'])
        print(scores[f'imbalance_by_class'])
        for label in range(num_classes):#Note to self: should change this later to the id
            #Metrics for Macrof1,MicroF1, individuals F1
            imb = scores[f'imbalance_by_class'][label]
            spec = scores[f'specificity_by_class'][label]
            sens = scores[f'recall_by_class'][label]
            real_case = 2* sens * (1+ imb) /((1+sens)*(1+imb)+ (1-spec)*(1-imb) ) 
            #real_case = 2* sens * (1+ imb) /(1 + sens + imb + sens*imb + 1 - spec - imb + spec*imb ) 
            #real_case = 2* sens * (1+ imb) /(2 + sens-spec + imb*(sens-spec) 
            #real_case = 2* sens * (1+ imb) /(2 + (1 + imb)(sens-spec) ) 
            # HIGH SPEC
            #real_case = 2* sens * smallnum /(2 + smallnum(sens-spec) ) 
            #real_case = low_num  
            #balanced_case = 2 * sens / (sens + 1 ) 
            # GENERAL CONCLUSIONS: 
            # 1. F1 BRINGS AN INHERENT NEGATIVE BIAS
            # 2. THIS INHERENT NEGATIVE BIAS COMES FROM 
            balanced_case = 2 * sens / (num_classes + sens + (1-num_classes) *spec) 
            scores[f'bias_f1_by_class'].append(real_case - balanced_case)
            #Microf1 needed calculations
            scores[f'microf1_terms_num'] += 2*sens
            scores[f'microf1_terms_denom_unbal'] += sens +1 + (1- spec)*(1-imb)/(1+imb)
            scores[f'microf1_terms_denom_bal'] += num_classes + sens + (1-num_classes) *spec
        scores[f'bias_f1macro'] = sum(scores[f'bias_f1_by_class{n}'])/num_classes
        scores[f'bias_f1weighted'] = sum([a*b for a,b in  zip(scores[f'bias_f1_by_class'],(scores[f'imbalance_by_class']+1)*2) ])
        scores[f'bias_f1micro'] =  \
            (scores[f'microf1_terms_num']/scores[f'microf1_terms_denom_unbal']) - \
            (scores[f'microf1_terms_num']/scores[f'microf1_terms_denom_bal'])
    write_metrics(scores)
    return scores



epochs = 8#3 
lr = 2e-5
bs= 16
physical_batch_size = 16#8
base_models = [
               'xlm-roberta-large', 
                'xlm-roberta-base',
               'xlnet-base-cased'
               ]
base_models = ["distilbert/distilbert-base-uncased"]
base_models = ["distilbert-base-uncased"]
#base_models = ["facebook/bart-large"]
#PARAM SEARCH 
for base_model in base_models: 
    for lr in  [1e-5,2e-5,5e-5]:
        for bs in [8,16,32]:
                        
            
            #TOKENIZING
            
            tokenizer = AutoTokenizer.from_pretrained(base_model)
            #token_start = '[CLS]'
            #token_e1= '[E1]'
            #token_e2= '[E2]'
            #token_sep= '[SEP]'
            
            
            def preprocess_function(e):
                return tokenizer(e["sentence"], e["e1_name"],e["e2_name"], truncation=True,padding=True)
            #print(dataset)
            #assert(False)
            
            tokenized_dataset = dataset.map(preprocess_function, batched=True)
            #print(tokenized_dataset['train'])
            print(tokenized_dataset['train'][0])
            print('-'*50)
            print(tokenized_dataset['validation'][0])
            #assert(False)

            tokenized_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
     
            #COLLATION    
            data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
            cur_epoch=1
            metrics = ['precision','recall','f1'] 
            
            #trainloader = DataLoader(tokenized_dataset['train'], batch_size=physical_batch_size, shuffle=True)
            
            #TRAINING
            
            model = AutoModelForSequenceClassification.from_pretrained(
                base_model, num_labels=len(id2label),problem_type="multi_label", id2label=id2label, label2id=label2id
            ).to('cuda' if torch.cuda.is_available() else 'cpu')
            model_name = f'{base_model}_lr{lr}_bs{bs}_epochs{epochs}'
            model_path = f"generated_models/{model_name}"
            cur_path = os.path.split(os.path.realpath(__file__))[0]
            datafile = os.path.join(cur_path, model_path)
            if not os.path.exists(datafile):
                Path(datafile).mkdir(parents=True,exist_ok=True)


            
            print(f'--------Training model {model_name}--------')
            training_args = TrainingArguments(
                output_dir=model_path,
                learning_rate=lr,
                gradient_accumulation_steps=int(bs/physical_batch_size),
                per_device_train_batch_size=int(physical_batch_size),
                per_device_eval_batch_size=int(physical_batch_size),
                num_train_epochs=epochs,
                weight_decay=0.01,
                evaluation_strategy="epoch",
                logging_strategy="epoch",
                save_strategy="no",
                metric_for_best_model='f1',
                load_best_model_at_end=False
            )
            
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset["train"],
                #train_dataset=trainloader,
                #eval_dataset=tokenized_dataset["validation"],
                #eval_dataset=tokenized_dataset["test"],
                eval_dataset=tokenized_dataset["train"],
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
            )
            
            trainer.evaluate()
            
            trainer.train()
            
            print('------TRAINING FINISHED----------')
            cur_path = os.path.split(os.path.realpath(__file__))[0]
            datafile = os.path.join(cur_path, model_path)
            if not os.path.exists(datafile):
                os.mkdir(datafile)
                #trainer.save_model(datafile)
            metric_name='f1'
            metrics_values = {f'val_{metric_name}':[],'val_loss':[],'tra_loss':[]}
            for metrics in trainer.state.log_history:
                if f'eval_{metric_name}' in metrics:
                    metrics_values['val_loss'].append(round(metrics['eval_loss'],3))
                    metrics_values[f'val_{metric_name}'].append(round(metrics[f'eval_{metric_name}'],3))
                elif 'loss' in metrics :
                    metrics_values['tra_loss'].append(round(metrics['loss'],3))
            
            def print_metrics():
                out = model_name + '\n'
                out += '\t'.join(['epoch'] + [str(i+1) for i in range(epochs)])
                for m in metrics_values:
                    out += '\n' + '\t'.join([m]+[str(i) for i in metrics_values[m]])
                eval_res = max(metrics_values[f'val_{metric_name}'])
                print(eval_res)
                out += f'\nBest {metric_name} on evaluation is {eval_res}'
                test_res = trainer.evaluate(tokenized_dataset["test"])
                print(test_res)
                out += f'\nBest {metric_name} on testing is {round(test_res[f"eval_{metric_name}"],3)}'
                return out
            
            with open(datafile+'/metrics.csv','w') as f:
                f.write(print_metrics())
            
            
            







