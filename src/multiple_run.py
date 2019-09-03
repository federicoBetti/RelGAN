import subprocess
import os

# Architecture Related
architecture = ['rmc_att_topic', 'rmc_att_topic', 'rmc_att_topic', 'rmc_att_topic']
topic_architecture = ['reuse_att_topic', 'standard', 'reuse_att_topic', 'standard']
gantype = ['standard', 'standard', 'standard', 'standard', 'standard']  # per ora funziona solo con questo il topic
gsteps = ['1', '2', '2', '2']
dsteps = ['5', '1', '2', '2']
npre_epochs = ['1', '400', '250', '150']
nadv_steps = ['10', '6000', '5000', '5000']
ntopic_pre_epochs = ['1', '500', '250', '50']
opt_type = ['adam', 'adam', 'adam', 'adam']
temperature = ['1000', '1000', '1000', '100']
d_lr = ['1e-4', '1e-4', '1e-4', '1e-4']
gadv_lr = ['1e-4', '1e-4', '1e-4', '1e-4']

# Topic Related
topic_number = ['9', '12', '3', '6']

# Memory Related
mem_slots = ['1', '1', '1', '1', '1', '1', '1', '1']
head_size = ['256', '256', '256', '256', '256', '256', '256', '256']
num_heads = ['2', '2', '2', '2', '2', '2', '2', '2']
seed = ['171', '172', '173', '174', '175', '176', '177', '178']

bs = '64'
gpre_lr = '1e-2'
hidden_dim = '32'
seq_len = '20'
dataset = 'emnlp_news'

gen_emb_dim = '32'
dis_emb_dim = '64'
num_rep = '64'
sn = False
decay = False
adapt = 'exp'
ntest = '20'

job_number = 1
configurations = []
for job_id in range(job_number):
    configurations.append([
        # Architecture
        '--topic',
        '--topic_number', topic_number[job_id],
        '--gf-dim', '64',
        '--df-dim', '64',
        '--g-architecture', architecture[job_id],
        '--d-architecture', architecture[job_id],
        '--topic-architecture', topic_architecture[job_id],
        '--gan-type', gantype[job_id],
        '--hidden-dim', hidden_dim,

        # Training
        '--gsteps', gsteps[job_id],
        '--dsteps', dsteps[job_id],
        '--npre-epochs', npre_epochs[job_id],
        '--nadv-steps', nadv_steps[job_id],
        '--n-topic-pre-epochs', ntopic_pre_epochs[job_id],
        '--ntest', ntest,
        '--d-lr', d_lr[job_id],
        '--gpre-lr', gpre_lr,
        '--gadv-lr', gadv_lr[job_id],
        '--batch-size', bs,
        '--log-dir', os.path.join('.', 'oracle', 'logs'),
        '--sample-dir', os.path.join('.', 'oracle', 'samples'),
        '--optimizer', opt_type[job_id],
        '--seed', seed[job_id],
        '--temperature', temperature[job_id],
        '--adapt', adapt,

        # evaluation
        '--nll-gen',
        # '--bleu',
        # '--selfbleu',
        # '--doc-embsim',
        '--KL',

        # relational memory
        '--mem-slots', mem_slots[job_id],
        '--head-size', head_size[job_id],
        '--num-heads', num_heads[job_id],

        # dataset
        '--dataset', dataset,
        '--vocab-size', '5000',
        '--start-token', '0',
        '--seq-len', seq_len,
        '--num-sentences', '10000',  # how many generated sentences to use per evaluation
        '--gen-emb-dim', gen_emb_dim,
        '--dis-emb-dim', dis_emb_dim,
        '--num-rep', num_rep,
        '--data-dir', os.path.join('.', 'data')
    ])

for configuration in configurations:
    subprocess.run(["python", "run.py"] + configuration, shell=False)
