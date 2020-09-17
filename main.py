import argparse
from prepare import *
from train_dkfn import *
from train_rnn import *
from train_gclstm import *

parser = argparse.ArgumentParser(description='traffic prediction')

# data
parser.add_argument('-dataset', type=str, default='metr_la', help='choose dataset to run [options: metr_la, seattle_loop]')

# model
parser.add_argument('-model', type=str, default='dkfn', help='choose model to train and test [options: rnn, lstm, gclstm, dkfn]')
args = parser.parse_args()

# load data
if args.dataset == 'metr_la':
    print("\nLoading metr_la data...")
    speed_matrix = pd.read_pickle('./METR_LA_Dataset/la_speed')
    A = np.load('./METR_LA_Dataset/METR_LA_A.npy')
elif args.dataset == 'seattle_loop':
    print("\nLoading seattle_loop data...")
    speed_matrix = pd.read_pickle('./Seattle_Loop_Dataset/sea_speed')
    A = np.load('./Seattle_Loop_Dataset/Loop_Seattle_A.npy')

print("\nPreparing data...")
train_dataloader, valid_dataloader, test_dataloader, max_speed = PrepareDataset(speed_matrix, BATCH_SIZE=64)

# model
if args.model == 'dkfn':
    print("\nTraining dkfn model...")
    dkfn, dkfn_loss = TrainDKFN(train_dataloader, valid_dataloader, A, K=3, num_epochs=100)
    print("\nTesting dkfn model...")
    results = TestDKFN(dkfn, test_dataloader, max_speed)

elif args.model == 'rnn':
    print("\nTraining rnn model...")
    rnn, rnn_loss = TrainRNN(train_dataloader, valid_dataloader, num_epochs=100)
    print("\nTesting rnn model...")
    results = TestRNN(rnn, test_dataloader, max_speed)

elif args.model == 'lstm':
    print("\nTraining lstm model...")
    lstm, lstm_loss = TrainLSTM(train_dataloader, valid_dataloader, num_epochs=100)
    print("\nTesting lstm model...")
    results = TestLSTM(lstm, test_dataloader, max_speed)

elif args.model == 'gclstm':
    print("\nTraining gclstm model...")
    gclstm, gclstm_loss = TrainGCLSTM(train_dataloader, valid_dataloader, A, K=3, num_epochs=100)
    print("\nTesting gclstm model...")
    results = TestGCLSTM(gclstm, test_dataloader, max_speed)


