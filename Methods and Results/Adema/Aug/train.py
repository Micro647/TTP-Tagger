import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pandas as pd
import numpy as np
import re
import json
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# ==================== 1. Load ATT&CK ID Mapping File ====================
mapping_df = pd.read_csv('2.csv', encoding='utf-8')

# Initialize mapping dictionaries: Name -> Label
tactic_mapping = {}
technique_mapping = {}

for _, row in mapping_df.iterrows():
    label = row['Label']
    name = row['Name']
    type_ = row['Type']
    
    if type_ == 'Tactic':
        tactic_mapping[name] = label
    else:  # Technique or Subtechnique
        technique_mapping[name] = label

# Extract ordered label lists from mapping
TACTIC = list(tactic_mapping.values())
TECHNIQUE = list(technique_mapping.values())

print(f"Number of tactics: {len(TACTIC)}")
print(f"Number of techniques/subtechniques: {len(TECHNIQUE)}")

# ==================== 2. Build Tactic-Technique Relationship Dictionary ====================
# Note: TACTICS_TECHNIQUES_RELATIONSHIP_DF should be defined here if available from original code
# This dictionary maps tactic IDs to their associated technique IDs
# If not available, it needs to be constructed from external ATT&CK data sources

# ==================== 3. Parse JSON Data and Construct Multi-label Vectors ====================
def parse_output(output_str):
    """
    Extract tactic, technique, and subtechnique names from output string.
    
    Args:
        output_str: String containing labels in format "Tactic:xxx, Technique:xxx, Subtechnique:xxx"
    
    Returns:
        tuple: (tactic_names_list, technique_names_list)
    """
    # Extract all Tactic labels
    tactic_pattern = r"Tactic:([^,\n]+)"
    tactics = re.findall(tactic_pattern, output_str)
    tactics = [t.strip() for t in tactics]
    
    # Extract all Technique labels
    tech_pattern = r"Technique:([^,\n]+)"
    techniques = re.findall(tech_pattern, output_str)
    techniques = [t.strip() for t in techniques]
    
    # Extract all Subtechnique labels
    sub_pattern = r"Subtechnique:([^,\n]+)"
    subtechniques = re.findall(sub_pattern, output_str)
    subtechniques = [t.strip() for t in subtechniques]
    
    # Merge techniques and subtechniques into a single list
    all_techniques = techniques + subtechniques
    
    return tactics, all_techniques

def create_multi_label_vector(names, mapping, label_list):
    """
    Create multi-label binary vector from label names.
    
    Args:
        names: List of label names (e.g., ['Persistence', 'Defense Evasion'])
        mapping: Dictionary mapping names to ATT&CK IDs
        label_list: Ordered list of all possible label IDs
    
    Returns:
        list: Binary vector of length len(label_list)
    """
    vec = [0] * len(label_list)
    for name in names:
        if name in mapping:
            label_id = mapping[name]
            if label_id in label_list:
                idx = label_list.index(label_id)
                vec[idx] = 1
            else:
                print(f"Warning: Label ID '{label_id}' not found in predefined label list")
        else:
            print(f"Warning: Cannot map name '{name}' to ATT&CK ID")
    return vec

# Load JSON dataset
json_file_path = 'your_data.json'  # Replace with actual file path
with open(json_file_path, 'r', encoding='utf-8') as f:
    json_data = json.load(f)

# Construct DataFrame from JSON records
records = []
for item in json_data:
    text = item['input']
    output_str = item['output']
    
    # Parse output labels
    tactic_names, technique_names = parse_output(output_str)
    
    # Create multi-label vectors
    tactic_vec = create_multi_label_vector(tactic_names, tactic_mapping, TACTIC)
    technique_vec = create_multi_label_vector(technique_names, technique_mapping, TECHNIQUE)
    
    records.append({
        'text': text,
        'tactic_label': tactic_vec,
        'technique_label': technique_vec,
        'raw_output': output_str  # Preserve for debugging
    })

df = pd.DataFrame(records)
print(f"Dataset size: {len(df)}")
print(f"Tactic label dimensionality: {len(TACTIC)}")
print(f"Technique label dimensionality: {len(TECHNIQUE)}")

# ==================== 4. Text Preprocessing Functions ====================
def get_wordnet_pos(tag):
    """Map POS tag to WordNet compatible POS tag."""
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return 'n'

def ioc_sub(text):
    """Substitute Indicators of Compromise (IOCs) with placeholder tokens."""
    def reg_handler(obj):
        s = obj.group(1)
        s = ' '.join(s.split('\\'))
        return s

    def file_handler(obj):
        s = obj.group(2)
        s = s.split('\\')[-1]
        return s
    
    # IPv4 address substitution
    text = re.sub(r'(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\.|\[\.\])){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\/([0-2][0-9]|3[0-2]|[0-9]))?', 'IPv4', text)
    text = re.sub(r'\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b', 'IP', text)
    
    # Vulnerability substitution
    text = re.sub(r'\b(CVE\-[0-9]{4}\-[0-9]{4,6})\b', 'CVE', text)
    text = re.sub(r'CVE-[0-9]{4}-[0-9]{4,6}', 'vulnerability', text)
    
    # Email substitution
    text = re.sub(r'\b([a-z][_a-z0-9-.]+@[a-z0-9-]+\.[a-z]+)\b', 'email', text)
    
    # Registry path substitution
    text = re.sub(r'\b((HKLM|HKCU|HKCR|HKU|HKCC)\\[\\A-Za-z0-9-_]+)\b', reg_handler, text)
    
    # File path substitution
    text = re.sub(r'\b([a-zA-Z]{1}:\\([0-9a-zA-Z_\.\-\/\\]+))\b', file_handler, text)
    
    # Hash substitution
    text = re.sub(r'\b([a-f0-9]{32}|[A-F0-9]{32})\b', 'MD5', text)
    text = re.sub(r'\b([a-f0-9]{40}|[A-F0-9]{40})\b', 'SHA1', text)
    text = re.sub(r'\b([a-f0-9]{64}|[A-F0-9]{64})\b', 'SHA256', text)
    text = re.sub(r'\d+:[A-Za-z0-9/+]+:[A-Za-z0-9/+]+', 'ssdeep', text)
    text = re.sub(r'\b[a-fA-F\d]{32}\b|\b[a-fA-F\d]{40}\b|\b[a-fA-F\d]{64}\b', 'hash', text)
    
    # URL substitution
    text = re.sub(r'h[tx][tx]ps?:[\\\/][\\\/](?:[0-9a-zA-Z_\.\-\/\\]|\[\.\])+', 'URL', text)
    
    # Domain substitution
    text = re.sub(r'((?:[0-9a-zA-Z_\-]+\.)+(?:(?!exe|dll)[a-z]{2,4}))', 'domain', text)
    
    # Remove other patterns
    text = re.sub(r'[a-fA-F0-9]{16}', '', text)
    text = re.sub(r'[0-9]{8}', '', text)
    text = re.sub(r'x[A-Fa-f0-9]{2}', '', text)
    
    return text

def rmstopword_and_lemmatize(text):
    """
    Remove stopwords and apply lemmatization with POS tagging.
    
    Args:
        text: Input text string
    
    Returns:
        str: Processed text with stopwords removed and words lemmatized
    """
    try:
        # Tokenize and remove stopwords
        token = [word for word in word_tokenize(text.lower()) if word not in stopwords.words('english')]
        # POS tagging
        tag = pos_tag(token)
        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        text = ' '.join(list(map(lambda x: lemmatizer.lemmatize(x[0], pos=get_wordnet_pos(x[1])), tag)))
    except:
        # Fallback processing if NLTK data is missing
        text = ' '.join([word for word in text.lower().split() if word not in stopwords.words('english')])
    return text

def preprocess(text):
    """
    Complete text preprocessing pipeline.
    
    Args:
        text: Raw input text
    
    Returns:
        str: Cleaned and normalized text
    """
    text = str(text)
    # Remove common patterns
    text = re.sub("\r\n", " ", text)
    text = re.sub('etc\.', '', text)
    text = re.sub('et al\.', '', text)
    text = re.sub('e\.g\.', '', text)
    text = re.sub('i\.e\.', '', text)
    # Remove citation markers
    text = re.sub(r'\[\d+\]', '', text)
    
    # Substitute IOCs
    text = ioc_sub(text)
    
    # Keep only alphanumeric characters and brackets
    text = re.sub(r'[^A-Za-z0-9_\[\]]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Apply stopword removal and lemmatization
    text = rmstopword_and_lemmatize(text)
    
    # Remove remaining brackets and single characters
    text = re.sub(r'[\[\]]', ' ', text)
    text = re.sub(r' [a-z0-9] ', '', text)
    
    return text

# Apply preprocessing to all texts
df['text_clean'] = df['text'].map(lambda t: preprocess(t))

# ==================== 5. Dataset Splitting ====================
train, test = train_test_split(df, test_size=0.2, random_state=2222)
validation, test = train_test_split(test, test_size=0.5, random_state=2222)

# Prepare technique classification datasets
df_te_train = df.iloc[train.index][['text_clean', 'technique_label']]
df_te_val = df.iloc[validation.index][['text_clean', 'technique_label']]
df_te_test = df.iloc[test.index][['text_clean', 'technique_label']]
df_te_train = df_te_train.rename(columns={'text_clean':'text', 'technique_label':'labels'})
df_te_val = df_te_val.rename(columns={'text_clean':'text', 'technique_label':'labels'})
df_te_test = df_te_test.rename(columns={'text_clean':'text', 'technique_label':'labels'})

# Prepare tactic classification datasets
df_ta_train = df.iloc[train.index][['text_clean', 'tactic_label']]
df_ta_val = df.iloc[validation.index][['text_clean', 'tactic_label']]
df_ta_test = df.iloc[test.index][['text_clean', 'tactic_label']]
df_ta_train = df_ta_train.rename(columns={'text_clean':'text', 'tactic_label':'labels'})
df_ta_val = df_ta_val.rename(columns={'text_clean':'text', 'tactic_label':'labels'})
df_ta_test = df_ta_test.rename(columns={'text_clean':'text', 'tactic_label':'labels'})

# ==================== 6. Model Training ====================
from simpletransformers.classification import MultiLabelClassificationModel, MultiLabelClassificationArgs
from sklearn.metrics import coverage_error, label_ranking_loss, label_ranking_average_precision_score
from sklearn.metrics import hamming_loss, precision_score, recall_score, f1_score, fbeta_score, accuracy_score

# Technique Multi-label Classification Model
te_multi_model_args = MultiLabelClassificationArgs()
te_multi_model_args.reprocess_input_data = True
te_multi_model_args.overwrite_output_dir = True
te_multi_model_args.evaluate_during_training = True
te_multi_model_args.use_multiprocessing = False
te_multi_model_args.use_multiprocessing_for_evaluation = False
te_multi_model_args.use_multiprocessed_decoding = False
te_multi_model_args.train_batch_size = 16
te_multi_model_args.eval_batch_size = 16
te_multi_model_args.num_train_epochs = 32
te_multi_model_args.use_early_stopping = True
te_multi_model_args.early_stopping_delta = 0.01
te_multi_model_args.early_stopping_metric = "eval_loss"
te_multi_model_args.early_stopping_metric_minimize = True
te_multi_model_args.early_stopping_patience = 10
te_multi_model_args.evaluate_during_training_steps = 1000
te_multi_model_args.learning_rate = 3e-5 
te_multi_model_args.output_dir = './outputs/technique_multi/'

# Initialize technique multi-label classifier
te_multi_model = MultiLabelClassificationModel(
    'distilbert',
    'distilbert-base-uncased',  # Pre-trained DistilBERT model
    use_cuda=True,
    num_labels=len(TECHNIQUE),
    args=te_multi_model_args,
)

# Train technique classification model
te_multi_model.train_model(df_te_train, eval_df=df_te_test, coverr=coverage_error, lrloss=label_ranking_loss)
te_result, te_model_outputs, te_wrong_predictions = te_multi_model.eval_model(df_te_test)

# Tactic Multi-label Classification Model
ta_multi_model_args = MultiLabelClassificationArgs()
ta_multi_model_args.reprocess_input_data = True
ta_multi_model_args.overwrite_output_dir = True
ta_multi_model_args.evaluate_during_training = True
ta_multi_model_args.manual_seed = 4
ta_multi_model_args.use_multiprocessing = False
ta_multi_model_args.use_multiprocessing_for_evaluation = False
ta_multi_model_args.use_multiprocessed_decoding = False
ta_multi_model_args.train_batch_size = 16
ta_multi_model_args.eval_batch_size = 16
ta_multi_model_args.num_train_epochs = 32
ta_multi_model_args.learning_rate = 5e-5
ta_multi_model_args.use_early_stopping = True
ta_multi_model_args.early_stopping_delta = 0.01
ta_multi_model_args.early_stopping_metric = "eval_loss"
ta_multi_model_args.early_stopping_metric_minimize = True
ta_multi_model_args.early_stopping_patience = 6
ta_multi_model_args.evaluate_during_training_steps = 1000
ta_multi_model_args.output_dir = './outputs/tactic_multi/'

# Initialize tactic multi-label classifier
ta_multi_model = MultiLabelClassificationModel(
    'distilbert',
    'distilbert-base-uncased',
    use_cuda=True,
    num_labels=len(TACTIC),
    args=ta_multi_model_args,
)

# Train tactic classification model
ta_multi_model.train_model(df_ta_train, eval_df=df_ta_test, coverr=coverage_error, lrloss=label_ranking_loss)
ta_result, ta_model_outputs, ta_wrong_predictions = ta_multi_model.eval_model(df_ta_test)

# ==================== 7. Model Evaluation ====================
# Evaluate technique classification model
te_predictions, te_model_outputs = te_multi_model.predict(list(df_te_test['text']))
te_true = np.array([row for row in df_te_test['labels']])

print('Technique Classification Results-----------------------------------------')
print('Coverage error: %f' % coverage_error(te_true, te_model_outputs))
print('Label Ranking Average Precision (LRAP): %f' % label_ranking_average_precision_score(te_true, te_model_outputs))
print('Label ranking loss: %f' % label_ranking_loss(te_true, te_model_outputs))

te_model_outputs_bi = te_predictions
print('Hamming loss: %f' % hamming_loss(te_true, te_model_outputs_bi))
print('Precision score (samples): %f' % precision_score(te_true, te_model_outputs_bi, average='samples', zero_division=0))
print('Precision score (macro): %f' % precision_score(te_true, te_model_outputs_bi, average='macro', zero_division=0))
print('Precision score (micro): %f' % precision_score(te_true, te_model_outputs_bi, average='micro', zero_division=0))
print('Recall score (samples): %f' % recall_score(te_true, te_model_outputs_bi, average='samples', zero_division=0))
print('Recall score (macro): %f' % recall_score(te_true, te_model_outputs_bi, average='macro', zero_division=0))
print('Recall score (micro): %f' % recall_score(te_true, te_model_outputs_bi, average='micro', zero_division=0))
print('F1 score (samples): %f' % f1_score(te_true, te_model_outputs_bi, average='samples', zero_division=0))
print('F1 score (macro): %f' % f1_score(te_true, te_model_outputs_bi, average='macro', zero_division=0))
print('F1 score (micro): %f' % f1_score(te_true, te_model_outputs_bi, average='micro', zero_division=0))
print('F0.5 score (samples): %f' % fbeta_score(te_true, te_model_outputs_bi, beta=0.5, average='samples', zero_division=0))
print('F0.5 score (macro): %f' % fbeta_score(te_true, te_model_outputs_bi, beta=0.5, average='macro', zero_division=0))
print('F0.5 score (micro): %f' % fbeta_score(te_true, te_model_outputs_bi, beta=0.5, average='micro', zero_division=0))
print('Accuracy score: %f' % accuracy_score(te_true, te_model_outputs_bi))

# Evaluate tactic classification model
ta_predictions, ta_model_outputs = ta_multi_model.predict(list(df_ta_test['text']))
ta_true = np.array([row for row in df.iloc[df_ta_test.index]['tactic_label']])

print('\nTactic Classification Results-----------------------------------------')
print('Coverage error: %f' % coverage_error(ta_true, ta_model_outputs))
print('Label Ranking Average Precision (LRAP): %f' % label_ranking_average_precision_score(ta_true, ta_model_outputs))
print('Label ranking loss: %f' % label_ranking_loss(ta_true, ta_model_outputs))

ta_model_outputs_bi = (ta_model_outputs > 0.5).astype(np.int_)
print('Hamming loss: %f' % hamming_loss(ta_true, ta_model_outputs_bi))
print('Precision score (samples): %f' % precision_score(ta_true, ta_model_outputs_bi, average='samples', zero_division=0))
print('Precision score (macro): %f' % precision_score(ta_true, ta_model_outputs_bi, average='macro', zero_division=0))
print('Precision score (micro): %f' % precision_score(ta_true, ta_model_outputs_bi, average='micro', zero_division=0))
print('Recall score (samples): %f' % recall_score(ta_true, ta_model_outputs_bi, average='samples', zero_division=0))
print('Recall score (macro): %f' % recall_score(ta_true, ta_model_outputs_bi, average='macro', zero_division=0))
print('Recall score (micro): %f' % recall_score(ta_true, ta_model_outputs_bi, average='micro', zero_division=0))
print('F1 score (samples): %f' % f1_score(ta_true, ta_model_outputs_bi, average='samples', zero_division=0))
print('F1 score (macro): %f' % f1_score(ta_true, ta_model_outputs_bi, average='macro', zero_division=0))
print('F1 score (micro): %f' % f1_score(ta_true, ta_model_outputs_bi, average='micro', zero_division=0))
print('F0.5 score (samples): %f' % fbeta_score(ta_true, ta_model_outputs_bi, beta=0.5, average='samples', zero_division=0))
print('F0.5 score (macro): %f' % fbeta_score(ta_true, ta_model_outputs_bi, beta=0.5, average='macro', zero_division=0))
print('F0.5 score (micro): %f' % fbeta_score(ta_true, ta_model_outputs_bi, beta=0.5, average='micro', zero_division=0))
print('Accuracy score: %f' % accuracy_score(ta_true, ta_model_outputs_bi))

# ==================== 8. Post-processing with Hierarchical Correction ====================
# Note: This section requires TACTICS_TECHNIQUES_RELATIONSHIP_DF to be defined
# The post-processing logic uses tactic predictions to refine technique predictions

print('\nPost-processing with Hierarchical Label Correction-------------------------')

# Initialize counters for post-processing analysis
ta_correct_true = {}
ta_correct_false = {}
sub_correct_true = {}
sub_correct_false = {}
highrate_correct_true = {}
highrate_correct_false = {}
all_true_mod = 0
all_false_mod = 0
all_true = 0
all_false = 0
true_origin = 0
false_origin = 0

te_modified = []
modified_ind = set()

for ind in range(len(df_te_test)):
    te_tmp = {}
    te_mask = {}
    te_pred = {}
    te_real = {}
    ta_real = {}
    ta_pred = {}

    # Thresholds for hierarchical correction
    ta_threshold = 0.01
    te_threshold = 0.25
    
    # Build tactic prediction dictionary
    for i, v in enumerate(ta_model_outputs[ind]):
        ta_pred[TACTIC[i]] = v
        # Create technique mask based on tactic predictions
        for te in TACTICS_TECHNIQUES_RELATIONSHIP_DF[TACTIC[i]]:
            try:
                te_mask[te] |= int(v > ta_threshold)
            except KeyError:
                te_mask[te] = int(v > ta_threshold)
    
    # Build technique prediction dictionary
    for te, v in list(zip(TECHNIQUE, te_model_outputs[ind])):
        te_pred[te] = v
    
    # Build ground truth dictionaries
    for te, v in list(zip(TECHNIQUE, te_true[ind])):
        te_real[te] = int(v)
    for ta, v in list(zip(TACTIC, ta_true[ind])):
        ta_real[ta] = int(v)

    # Apply hierarchical correction
    tp = fp = tn = fn = 0
    tp_ = fp_ = tn_ = fn_ = 0
    
    for te in TECHNIQUE:
        try:
            if te_tmp[te]:
                continue
        except KeyError:
            # High confidence predictions (>0.95) bypass correction
            if te_pred[te] > 0.95:
                te_mask[te] = 1
                if te_real[te]:
                    try:
                        highrate_correct_true[te] += 1
                    except KeyError:
                        highrate_correct_true[te] = 1
                else:
                    try:
                        highrate_correct_false[te] += 1
                    except KeyError:
                        highrate_correct_false[te] = 1
                    
            te_tmp[te] = int(te_pred[te] >= te_threshold) & te_mask[te]
        
        # Apply threshold-based classification with tactic constraints
        if te_pred[te] >= te_threshold:
            if te_real[te]:
                tp += 1
                if te_mask[te]:
                    tp_ += 1
                    
                    # Propagate to parent technique if applicable
                    if len(te.split('.')) > 1:
                        te_parent = te.split('.')[0]
                        try:
                            if te_tmp[te_parent]:
                                pass
                            else:
                                raise KeyError
                        except KeyError:
                            te_tmp[te_parent] = 1
                            
                            if te_real[te_parent]:
                                if te_pred[te_parent] < te_threshold:
                                    try:
                                        sub_correct_true[te_parent] += 1
                                    except KeyError:
                                        sub_correct_true[te_parent] = 1
                                    status = 'True'
                                else:
                                    status = 'Useless'
                            else:
                                try:
                                    sub_correct_false[te_parent] += 1
                                except KeyError:
                                    sub_correct_false[te_parent] = 1
                                status = 'False'
                            if status != 'Useless':
                                print(f'{ind}: {te_parent}: {te}, parent_real={te_real[te_parent]} parent_pred={te_pred[te_parent]}, child_real={te_real[te]} child_pred={te_pred[te]} -> {status}')
                else:
                    fn_ += 1
                    # Log false negatives due to tactic constraints
                    for ta in TACTICS_TECHNIQUES_RELATIONSHIP_DF:
                        if te in TACTICS_TECHNIQUES_RELATIONSHIP_DF[ta].unique():
                            print(f'{ind}: {ta}: {te}, ta_real={ta_real[ta]} ta_pred={ta_pred[ta]}, te_real={te_real[te]} te_mask={te_mask[te]} te_pred={te_pred[te]} -> false')
                            break
                    try:
                        ta_correct_false[te] += 1
                    except KeyError:
                        ta_correct_false[te] = 1
            else:
                fp += 1
                if te_mask[te]:
                    fp_ += 1
                else:
                    tn_ += 1
                    # Log true negatives due to tactic constraints
                    for ta in TACTICS_TECHNIQUES_RELATIONSHIP_DF:
                        if te in TACTICS_TECHNIQUES_RELATIONSHIP_DF[ta].unique():
                            print(f'{ind}: {ta}: {te}, ta_real={ta_real[ta]} ta_pred={ta_pred[ta]}, te_real={te_real[te]} te_mask={te_mask[te]} te_pred={te_pred[te]} -> true')
                            break
                    try:
                        ta_correct_true[te] += 1
                    except KeyError:
                        ta_correct_true[te] = 1
        else:
            if te_real[te]:
                fn += 1
                fn_ += 1
            else:
                tn += 1
                tn_ += 1
    
    # Aggregate modification statistics
    true_mod = 0
    false_mod = 0
    true = 0
    false = 0
    true_ori = 0
    false_ori = 0
    
    for te in TECHNIQUE:
        if te_real[te]:
            if te_pred[te] >= 0.5:
                true_ori += 1
                if te_tmp[te]:
                    true += 1
                else:
                    false += 1
                    false_mod += 1
            else:
                false_ori += 1
                if te_tmp[te]:
                    true += 1
                    true_mod += 1
                else:
                    false += 1
    
    all_true_mod += true_mod
    all_false_mod += false_mod
    all_true += true
    all_false += false
    true_origin += true_ori
    false_origin += false_ori

    # Store modified predictions
    te_tmp = [te_tmp[te] for te in TECHNIQUE]
    te_modified.append(te_tmp)

# Evaluate post-processing improvements
te_model_outputs_bi = (te_model_outputs > 0.5).astype(np.int_)
print('\nPost-processing Results (Original -> Modified):')
print('Hamming loss: %f -> %f' % (hamming_loss(te_true, te_model_outputs_bi), hamming_loss(te_true, te_modified)))
print('Precision score (samples): %f -> %f' % (precision_score(te_true, te_model_outputs_bi, average='samples', zero_division=0), precision_score(te_true, te_modified, average='samples', zero_division=0)))
print('Precision score (macro): %f -> %f' % (precision_score(te_true, te_model_outputs_bi, average='macro', zero_division=0), precision_score(te_true, te_modified, average='macro', zero_division=0)))
print('Precision score (micro): %f -> %f' % (precision_score(te_true, te_model_outputs_bi, average='micro', zero_division=0), precision_score(te_true, te_modified, average='micro', zero_division=0)))
print('Recall score (samples): %f -> %f' % (recall_score(te_true, te_model_outputs_bi, average='samples', zero_division=0), recall_score(te_true, te_modified, average='samples', zero_division=0)))
print('Recall score (macro): %f -> %f' % (recall_score(te_true, te_model_outputs_bi, average='macro', zero_division=0), recall_score(te_true, te_modified, average='macro', zero_division=0)))
print('Recall score (micro): %f -> %f' % (recall_score(te_true, te_model_outputs_bi, average='micro', zero_division=0), recall_score(te_true, te_modified, average='micro', zero_division=0)))
print('F1 score (samples): %f -> %f' % (f1_score(te_true, te_model_outputs_bi, average='samples', zero_division=0), f1_score(te_true, te_modified, average='samples', zero_division=0)))
print('F1 score (macro): %f -> %f' % (f1_score(te_true, te_model_outputs_bi, average='macro', zero_division=0), f1_score(te_true, te_modified, average='macro', zero_division=0)))
print('F1 score (micro): %f -> %f' % (f1_score(te_true, te_model_outputs_bi, average='micro', zero_division=0), f1_score(te_true, te_modified, average='micro', zero_division=0)))
print('F0.5 score (samples): %f -> %f' % (fbeta_score(te_true, te_model_outputs_bi, beta=0.5, average='samples', zero_division=0), fbeta_score(te_true, te_modified, beta=0.5, average='samples', zero_division=0)))
print('F0.5 score (macro): %f -> %f' % (fbeta_score(te_true, te_model_outputs_bi, beta=0.5, average='macro', zero_division=0), fbeta_score(te_true, te_modified, beta=0.5, average='macro', zero_division=0)))
print('F0.5 score (micro): %f -> %f' % (fbeta_score(te_true, te_model_outputs_bi, beta=0.5, average='micro', zero_division=0), fbeta_score(te_true, te_modified, beta=0.5, average='micro', zero_division=0)))
print('Accuracy score: %f -> %f' % (accuracy_score(te_true, te_model_outputs_bi), accuracy_score(te_true, te_modified)))