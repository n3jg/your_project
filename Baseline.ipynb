{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 60,
   "metadata": {},
   "outputs": [],
   "source": [
    "# path to your train/test/meta folders\n",
    "DATA_PATH = '../'\n",
    "\n",
    "# names of valuable files/folders\n",
    "train_meta_fname = 'train.csv'\n",
    "test_meta_fname = 'sample_submission.csv'\n",
    "train_data_folder = 'train'\n",
    "test_data_folder = 'test'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import os\n",
    "import pandas as pd\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.nn.functional as F\n",
    "from torch.utils.data import Dataset\n",
    "from torch.utils.data import DataLoader\n",
    "import torchaudio\n",
    "import torchvision\n",
    "from torchaudio import transforms\n",
    "from efficientnet_pytorch import EfficientNet\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import f1_score\n",
    "from tqdm import tqdm"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "metadata": {},
   "outputs": [],
   "source": [
    "# set seeds\n",
    "import random\n",
    "import numpy as np\n",
    "\n",
    "random.seed(42)\n",
    "np.random.seed(42)\n",
    "torch.manual_seed(42)\n",
    "torch.cuda.manual_seed(42)\n",
    "torch.backends.cudnn.deterministic = True"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>fname</th>\n",
       "      <th>label</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>8bcbcc394ba64fe85ed4.wav</td>\n",
       "      <td>Finger_snapping</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>00d77b917e241afa06f1.wav</td>\n",
       "      <td>Squeak</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                      fname            label\n",
       "0  8bcbcc394ba64fe85ed4.wav  Finger_snapping\n",
       "1  00d77b917e241afa06f1.wav           Squeak"
      ]
     },
     "execution_count": 74,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_train = pd.read_csv(os.path.join(DATA_PATH, train_meta_fname))\n",
    "df_test = pd.read_csv(os.path.join(DATA_PATH, test_meta_fname))\n",
    "df_train.head(2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "41\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>fname</th>\n",
       "      <th>label</th>\n",
       "      <th>label_encoded</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>8bcbcc394ba64fe85ed4.wav</td>\n",
       "      <td>Finger_snapping</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>00d77b917e241afa06f1.wav</td>\n",
       "      <td>Squeak</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>17bb93b73b8e79234cb3.wav</td>\n",
       "      <td>Electric_piano</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>7d5c7a40a936136da55e.wav</td>\n",
       "      <td>Harmonica</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>17e0ee7565a33d6c2326.wav</td>\n",
       "      <td>Snare_drum</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                      fname            label  label_encoded\n",
       "0  8bcbcc394ba64fe85ed4.wav  Finger_snapping              0\n",
       "1  00d77b917e241afa06f1.wav           Squeak              1\n",
       "2  17bb93b73b8e79234cb3.wav   Electric_piano              2\n",
       "3  7d5c7a40a936136da55e.wav        Harmonica              3\n",
       "4  17e0ee7565a33d6c2326.wav       Snare_drum              4"
      ]
     },
     "execution_count": 75,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "n_classes = df_train.label.nunique()\n",
    "print(n_classes)\n",
    "classes_dict = {cl:i for i,cl in enumerate(df_train.label.unique())}\n",
    "df_train['label_encoded'] = df_train.label.map(classes_dict)\n",
    "df_train.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "metadata": {},
   "outputs": [],
   "source": [
    "# https://github.com/lukemelas/EfficientNet-PyTorch\n",
    "class BaseLineModel(nn.Module):\n",
    "    \n",
    "    def __init__(self, sample_rate=16000, n_classes=41):\n",
    "        super().__init__()\n",
    "        self.ms = torchaudio.transforms.MelSpectrogram(sample_rate)\n",
    "#         self.bn1 = nn.BatchNorm2d(1)\n",
    "        \n",
    "        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, padding=1)\n",
    "        self.cnn3 = nn.Conv2d(in_channels=10, out_channels=3, kernel_size=3, padding=1)\n",
    "        \n",
    "        self.features = EfficientNet.from_pretrained('efficientnet-b0')\n",
    "        # use it as features\n",
    "#         for param in self.features.parameters():\n",
    "#             param.requires_grad = False\n",
    "            \n",
    "        self.lin1 = nn.Linear(1000, 333)\n",
    "        \n",
    "        self.lin2 = nn.Linear(333, 111)\n",
    "                \n",
    "        self.lin3 = nn.Linear(111, n_classes)\n",
    "        \n",
    "    def forward(self, x):\n",
    "        x = self.ms(x)\n",
    "#         x = self.bn1(x)\n",
    "                \n",
    "        x = F.relu(self.cnn1(x))\n",
    "        x = F.relu(self.cnn3(x))\n",
    "        \n",
    "        x = self.features(x)\n",
    "\n",
    "        x = x.view(x.shape[0], -1)\n",
    "        x = F.relu(x)\n",
    "        \n",
    "        x = F.relu(self.lin1(x))\n",
    "        x = F.relu(self.lin2(x))\n",
    "        x = self.lin3(x)\n",
    "        return x\n",
    "    \n",
    "    def inference(self, x):\n",
    "        x = self.forward(x)\n",
    "        x = F.softmax(x)\n",
    "        return x"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "metadata": {},
   "outputs": [],
   "source": [
    "def sample_or_pad(waveform, wav_len=32000):\n",
    "    m, n = waveform.shape\n",
    "    if n < wav_len:\n",
    "        padded_wav = torch.zeros(1, wav_len)\n",
    "        padded_wav[:, :n] = waveform\n",
    "        return padded_wav\n",
    "    elif n > wav_len:\n",
    "        offset = np.random.randint(0, n - wav_len)\n",
    "        sampled_wav = waveform[:, offset:offset+wav_len]\n",
    "        return sampled_wav\n",
    "    else:\n",
    "        return waveform\n",
    "        \n",
    "class EventDetectionDataset(Dataset):\n",
    "    def __init__(self, data_path, x, y=None):\n",
    "        self.x = x\n",
    "        self.y = y\n",
    "        self.data_path = data_path\n",
    "    \n",
    "    def __len__(self):\n",
    "        return len(self.x)\n",
    "\n",
    "    def __getitem__(self, idx):\n",
    "        path2wav = os.path.join(self.data_path, self.x[idx])\n",
    "        waveform, sample_rate = torchaudio.load(path2wav, normalization=True)\n",
    "        waveform = sample_or_pad(waveform)\n",
    "        if self.y is not None:\n",
    "            return waveform, self.y[idx]\n",
    "        return waveform"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_val, y_train, y_val = train_test_split(df_train.fname.values, df_train.label_encoded.values, \n",
    "                                                  test_size=0.2, random_state=42)\n",
    "train_loader = DataLoader(\n",
    "                        EventDetectionDataset(os.path.join(DATA_PATH, train_data_folder), X_train, y_train),\n",
    "                        batch_size=41\n",
    "                )\n",
    "val_loader = DataLoader(\n",
    "                        EventDetectionDataset(os.path.join(DATA_PATH, train_data_folder), X_val, y_val),\n",
    "                        batch_size=41\n",
    "                )\n",
    "test_loader = DataLoader(\n",
    "                        EventDetectionDataset(os.path.join(DATA_PATH, test_data_folder), df_test.fname.values, None),\n",
    "                        batch_size=41, shuffle=False\n",
    "                )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 79,
   "metadata": {},
   "outputs": [],
   "source": [
    "def eval_model(model, eval_dataset):\n",
    "    model.eval()\n",
    "    forecast, true_labs = [], []\n",
    "    with torch.no_grad():\n",
    "        for wavs, labs in tqdm(eval_dataset):\n",
    "            wavs, labs = wavs.cuda(), labs.detach().numpy()\n",
    "            true_labs.append(labs)\n",
    "            outputs = model.inference(wavs)\n",
    "            \n",
    "            outputs = outputs.detach().cpu().numpy().argmax(axis=1)\n",
    "            forecast.append(outputs)\n",
    "    forecast = [x for sublist in forecast for x in sublist]\n",
    "    true_labs = [x for sublist in true_labs for x in sublist]\n",
    "    return f1_score(forecast, true_labs, average='macro')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 80,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Loaded pretrained weights for efficientnet-b0\n"
     ]
    }
   ],
   "source": [
    "criterion = nn.CrossEntropyLoss()\n",
    "model = BaseLineModel()\n",
    "model = model.cuda()\n",
    "lr = 1e-3\n",
    "\n",
    "optimizer = torch.optim.Adam(model.parameters(), lr=lr)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 81,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:13<00:00,  8.28it/s]\n",
      "  0%|          | 0/28 [00:00<?, ?it/s]/home/ilin-a@ad.speechpro.com/miniconda3/envs/itmo-baseline/lib/python3.7/site-packages/ipykernel_launcher.py:60: UserWarning: Implicit dimension choice for softmax has been deprecated. Change the call to include dim=X as an argument.\n",
      "100%|██████████| 28/28 [00:01<00:00, 16.32it/s]\n",
      "100%|██████████| 111/111 [00:07<00:00, 15.62it/s]\n",
      "  1%|          | 1/111 [00:00<00:14,  7.63it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 0, f1_test: 0.10629352612610805, f1_train: 0.11949593809290951\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:13<00:00,  8.29it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 15.62it/s]\n",
      "100%|██████████| 111/111 [00:07<00:00, 15.62it/s]\n",
      "  0%|          | 0/111 [00:00<?, ?it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 1, f1_test: 0.2285954430090747, f1_train: 0.2344379529881226\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:13<00:00,  7.95it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 16.20it/s]\n",
      "100%|██████████| 111/111 [00:06<00:00, 15.89it/s]\n",
      "  0%|          | 0/111 [00:00<?, ?it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 2, f1_test: 0.25355097340148225, f1_train: 0.29173411590335957\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:13<00:00,  8.11it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 16.18it/s]\n",
      "100%|██████████| 111/111 [00:07<00:00, 15.45it/s]\n",
      "  0%|          | 0/111 [00:00<?, ?it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 3, f1_test: 0.3561665093600878, f1_train: 0.4097082985139217\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:13<00:00,  8.31it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 16.06it/s]\n",
      "100%|██████████| 111/111 [00:07<00:00, 15.54it/s]\n",
      "  1%|          | 1/111 [00:00<00:14,  7.59it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 4, f1_test: 0.3494387466921597, f1_train: 0.45181671057461237\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:13<00:00,  8.18it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 15.34it/s]\n",
      "100%|██████████| 111/111 [00:07<00:00, 15.62it/s]\n",
      "  0%|          | 0/111 [00:00<?, ?it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 5, f1_test: 0.4328167708131114, f1_train: 0.5271681413449881\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:13<00:00,  8.18it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 17.17it/s]\n",
      "100%|██████████| 111/111 [00:07<00:00, 14.95it/s]\n",
      "  0%|          | 0/111 [00:00<?, ?it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 6, f1_test: 0.4746482252679639, f1_train: 0.5223790932627627\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:13<00:00,  8.22it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 15.77it/s]\n",
      "100%|██████████| 111/111 [00:07<00:00, 15.53it/s]\n",
      "  0%|          | 0/111 [00:00<?, ?it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 7, f1_test: 0.5529561680590374, f1_train: 0.6627944015908095\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:13<00:00,  8.09it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 16.57it/s]\n",
      "100%|██████████| 111/111 [00:07<00:00, 15.58it/s]\n",
      "  0%|          | 0/111 [00:00<?, ?it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 8, f1_test: 0.5610194290738628, f1_train: 0.696918293914471\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:13<00:00,  8.27it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 16.09it/s]\n",
      "100%|██████████| 111/111 [00:07<00:00, 15.43it/s]\n",
      "  1%|          | 1/111 [00:00<00:14,  7.62it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 9, f1_test: 0.48521044792437856, f1_train: 0.6039715630375571\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:13<00:00,  8.16it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 15.54it/s]\n",
      "100%|██████████| 111/111 [00:06<00:00, 16.05it/s]\n",
      "  1%|          | 1/111 [00:00<00:14,  7.65it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 10, f1_test: 0.53396085877329, f1_train: 0.690427261583096\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:13<00:00,  8.14it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 17.08it/s]\n",
      "100%|██████████| 111/111 [00:07<00:00, 15.78it/s]\n",
      "  0%|          | 0/111 [00:00<?, ?it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 11, f1_test: 0.5996557969970892, f1_train: 0.7674035868883995\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:13<00:00,  8.29it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 16.29it/s]\n",
      "100%|██████████| 111/111 [00:07<00:00, 15.23it/s]\n",
      "  1%|          | 1/111 [00:00<00:19,  5.71it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 12, f1_test: 0.5317764072598671, f1_train: 0.6863034606639081\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:13<00:00,  8.32it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 15.80it/s]\n",
      "100%|██████████| 111/111 [00:07<00:00, 15.14it/s]\n",
      "  1%|          | 1/111 [00:00<00:14,  7.63it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 13, f1_test: 0.5256393615720969, f1_train: 0.6747247578159984\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:13<00:00,  8.24it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 16.77it/s]\n",
      "100%|██████████| 111/111 [00:07<00:00, 15.23it/s]\n",
      "  1%|          | 1/111 [00:00<00:14,  7.63it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 14, f1_test: 0.5418535358430485, f1_train: 0.6913837244584565\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:13<00:00,  8.15it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 15.75it/s]\n",
      "100%|██████████| 111/111 [00:07<00:00, 15.53it/s]\n",
      "  0%|          | 0/111 [00:00<?, ?it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 15, f1_test: 0.6226587230729366, f1_train: 0.8014973351379933\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:13<00:00,  8.02it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 14.33it/s]\n",
      "100%|██████████| 111/111 [00:07<00:00, 15.21it/s]\n",
      "  1%|          | 1/111 [00:00<00:14,  7.69it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 16, f1_test: 0.42308197459757096, f1_train: 0.5504070654979158\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:13<00:00,  8.32it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 17.15it/s]\n",
      "100%|██████████| 111/111 [00:07<00:00, 15.73it/s]\n",
      "  1%|          | 1/111 [00:00<00:14,  7.58it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 17, f1_test: 0.49867998540384506, f1_train: 0.658857005069475\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:13<00:00,  8.20it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 17.15it/s]\n",
      "100%|██████████| 111/111 [00:07<00:00, 15.66it/s]\n",
      "  1%|          | 1/111 [00:00<00:14,  7.53it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 18, f1_test: 0.5579497821999947, f1_train: 0.7319272730225758\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:13<00:00,  8.26it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 17.11it/s]\n",
      "100%|██████████| 111/111 [00:06<00:00, 16.30it/s]\n",
      "  1%|          | 1/111 [00:00<00:15,  7.06it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 19, f1_test: 0.5571325006232938, f1_train: 0.6957903774635286\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:13<00:00,  8.15it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 16.62it/s]\n",
      "100%|██████████| 111/111 [00:07<00:00, 15.64it/s]\n",
      "  0%|          | 0/111 [00:00<?, ?it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 20, f1_test: 0.46955998060031356, f1_train: 0.6558132821268612\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:14<00:00,  7.49it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 14.14it/s]\n",
      "100%|██████████| 111/111 [00:08<00:00, 13.50it/s]\n",
      "  1%|          | 1/111 [00:00<00:15,  7.01it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 21, f1_test: 0.48981598954550704, f1_train: 0.650733932483711\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:13<00:00,  8.19it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 17.03it/s]\n",
      "100%|██████████| 111/111 [00:07<00:00, 14.33it/s]\n",
      "  1%|          | 1/111 [00:00<00:14,  7.53it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 22, f1_test: 0.48420795722671806, f1_train: 0.6607939313215059\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:14<00:00,  7.80it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 15.58it/s]\n",
      "100%|██████████| 111/111 [00:07<00:00, 14.05it/s]\n",
      "  1%|          | 1/111 [00:00<00:14,  7.66it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 23, f1_test: 0.4508735701586984, f1_train: 0.5896833421726222\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:13<00:00,  8.09it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 17.32it/s]\n",
      "100%|██████████| 111/111 [00:07<00:00, 14.76it/s]\n",
      "  1%|          | 1/111 [00:00<00:14,  7.58it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 24, f1_test: 0.571814169964218, f1_train: 0.7611057880835792\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:13<00:00,  8.04it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 16.14it/s]\n",
      "100%|██████████| 111/111 [00:07<00:00, 14.59it/s]\n",
      "  0%|          | 0/111 [00:00<?, ?it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 25, f1_test: 0.5233877218544075, f1_train: 0.7131283965364282\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:14<00:00,  7.49it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 15.80it/s]\n",
      "100%|██████████| 111/111 [00:07<00:00, 13.96it/s]\n",
      "  1%|          | 1/111 [00:00<00:14,  7.64it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 26, f1_test: 0.5341731840891024, f1_train: 0.7076250982567919\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:14<00:00,  7.90it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 15.27it/s]\n",
      "100%|██████████| 111/111 [00:07<00:00, 15.26it/s]\n",
      "  1%|          | 1/111 [00:00<00:19,  5.65it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 27, f1_test: 0.5473086976098137, f1_train: 0.7375620636068481\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:14<00:00,  7.83it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 14.75it/s]\n",
      "100%|██████████| 111/111 [00:07<00:00, 14.54it/s]\n",
      "  1%|          | 1/111 [00:00<00:16,  6.86it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 28, f1_test: 0.5083482172203078, f1_train: 0.6905922529062383\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:14<00:00,  7.76it/s]\n",
      "100%|██████████| 28/28 [00:02<00:00, 13.70it/s]\n",
      "100%|██████████| 111/111 [00:08<00:00, 13.54it/s]\n",
      "  1%|          | 1/111 [00:00<00:14,  7.66it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 29, f1_test: 0.5513084368952619, f1_train: 0.7726534498333878\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:14<00:00,  7.77it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 15.59it/s]\n",
      "100%|██████████| 111/111 [00:08<00:00, 13.79it/s]\n",
      "  1%|          | 1/111 [00:00<00:16,  6.82it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 30, f1_test: 0.594515325055727, f1_train: 0.8269128506549175\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:14<00:00,  7.64it/s]\n",
      "100%|██████████| 28/28 [00:02<00:00, 12.37it/s]\n",
      "100%|██████████| 111/111 [00:07<00:00, 14.09it/s]\n",
      "  1%|          | 1/111 [00:00<00:20,  5.30it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 31, f1_test: 0.5583317614606507, f1_train: 0.7563482798684882\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:14<00:00,  7.63it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 14.78it/s]\n",
      "100%|██████████| 111/111 [00:07<00:00, 14.12it/s]\n",
      "  1%|          | 1/111 [00:00<00:15,  6.97it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 32, f1_test: 0.5941509798244281, f1_train: 0.8002843711761052\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:14<00:00,  7.52it/s]\n",
      "100%|██████████| 28/28 [00:02<00:00, 13.02it/s]\n",
      "100%|██████████| 111/111 [00:08<00:00, 13.76it/s]\n",
      "  1%|          | 1/111 [00:00<00:14,  7.63it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 33, f1_test: 0.5140710680051351, f1_train: 0.6838671653410366\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:14<00:00,  7.71it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 15.58it/s]\n",
      "100%|██████████| 111/111 [00:08<00:00, 13.61it/s]\n",
      "  1%|          | 1/111 [00:00<00:18,  5.89it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 34, f1_test: 0.5970381640976719, f1_train: 0.8124793248735787\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:14<00:00,  7.67it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 15.37it/s]\n",
      "100%|██████████| 111/111 [00:07<00:00, 14.35it/s]\n",
      "  1%|          | 1/111 [00:00<00:15,  6.95it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 35, f1_test: 0.5479606898586282, f1_train: 0.7397387901488383\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:14<00:00,  7.71it/s]\n",
      "100%|██████████| 28/28 [00:02<00:00, 13.28it/s]\n",
      "100%|██████████| 111/111 [00:07<00:00, 14.02it/s]\n",
      "  1%|          | 1/111 [00:00<00:14,  7.54it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 36, f1_test: 0.6114975741330437, f1_train: 0.8352293229768932\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:14<00:00,  7.61it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 14.37it/s]\n",
      "100%|██████████| 111/111 [00:07<00:00, 14.05it/s]\n",
      "  1%|          | 1/111 [00:00<00:17,  6.18it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 37, f1_test: 0.502708376143018, f1_train: 0.6816692501871874\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:14<00:00,  7.85it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 16.58it/s]\n",
      "100%|██████████| 111/111 [00:07<00:00, 14.10it/s]\n",
      "  1%|          | 1/111 [00:00<00:17,  6.12it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 38, f1_test: 0.5708448016964366, f1_train: 0.7498244171365385\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:14<00:00,  7.84it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 15.39it/s]\n",
      "100%|██████████| 111/111 [00:08<00:00, 13.47it/s]\n",
      "  1%|          | 1/111 [00:00<00:15,  7.26it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 39, f1_test: 0.5879043397739223, f1_train: 0.7929470964158711\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:14<00:00,  7.78it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 15.33it/s]\n",
      "100%|██████████| 111/111 [00:08<00:00, 13.75it/s]\n",
      "  1%|          | 1/111 [00:00<00:15,  7.11it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 40, f1_test: 0.5062550721992987, f1_train: 0.7154368513125894\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:14<00:00,  7.53it/s]\n",
      "100%|██████████| 28/28 [00:02<00:00, 13.71it/s]\n",
      "100%|██████████| 111/111 [00:08<00:00, 13.63it/s]\n",
      "  0%|          | 0/111 [00:00<?, ?it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 41, f1_test: 0.6252973186176202, f1_train: 0.8505659057903101\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:14<00:00,  7.82it/s]\n",
      "100%|██████████| 28/28 [00:02<00:00, 13.66it/s]\n",
      "100%|██████████| 111/111 [00:08<00:00, 13.82it/s]\n",
      "  1%|          | 1/111 [00:00<00:16,  6.87it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 42, f1_test: 0.6111373805075606, f1_train: 0.7887227495054853\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:14<00:00,  7.73it/s]\n",
      "100%|██████████| 28/28 [00:02<00:00, 13.23it/s]\n",
      "100%|██████████| 111/111 [00:07<00:00, 14.82it/s]\n",
      "  1%|          | 1/111 [00:00<00:16,  6.74it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 43, f1_test: 0.5286643820007692, f1_train: 0.7337212960601662\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:15<00:00,  7.30it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 14.39it/s]\n",
      "100%|██████████| 111/111 [00:08<00:00, 13.76it/s]\n",
      "  1%|          | 1/111 [00:00<00:14,  7.42it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 44, f1_test: 0.5956040081370013, f1_train: 0.8300039085580488\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:14<00:00,  7.83it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 15.77it/s]\n",
      "100%|██████████| 111/111 [00:07<00:00, 15.53it/s]\n",
      "  1%|          | 1/111 [00:00<00:14,  7.54it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 45, f1_test: 0.5644468974243524, f1_train: 0.7662344297044733\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:13<00:00,  8.08it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 15.67it/s]\n",
      "100%|██████████| 111/111 [00:07<00:00, 15.71it/s]\n",
      "  1%|          | 1/111 [00:00<00:14,  7.59it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 46, f1_test: 0.5497737424438712, f1_train: 0.7428294431954231\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:13<00:00,  8.03it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 17.19it/s]\n",
      "100%|██████████| 111/111 [00:07<00:00, 15.10it/s]\n",
      "  0%|          | 0/111 [00:00<?, ?it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 47, f1_test: 0.667015937016888, f1_train: 0.9370448561961691\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:13<00:00,  8.03it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 16.56it/s]\n",
      "100%|██████████| 111/111 [00:07<00:00, 14.86it/s]\n",
      "  1%|          | 1/111 [00:00<00:14,  7.54it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 48, f1_test: 0.59345882150071, f1_train: 0.8171683320930807\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:13<00:00,  8.17it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 16.25it/s]\n",
      "100%|██████████| 111/111 [00:07<00:00, 15.16it/s]\n",
      "  1%|          | 1/111 [00:00<00:14,  7.59it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 49, f1_test: 0.5879719740163387, f1_train: 0.8222908617749788\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:13<00:00,  8.12it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 15.42it/s]\n",
      "100%|██████████| 111/111 [00:07<00:00, 15.07it/s]\n",
      "  1%|          | 1/111 [00:00<00:14,  7.52it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 50, f1_test: 0.5819638496946737, f1_train: 0.831340747459372\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:14<00:00,  7.86it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 17.25it/s]\n",
      "100%|██████████| 111/111 [00:07<00:00, 15.11it/s]\n",
      "  1%|          | 1/111 [00:00<00:14,  7.67it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 51, f1_test: 0.5602648771549301, f1_train: 0.7694531491527006\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:13<00:00,  8.00it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 15.81it/s]\n",
      "100%|██████████| 111/111 [00:07<00:00, 15.00it/s]\n",
      "  1%|          | 1/111 [00:00<00:14,  7.64it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 52, f1_test: 0.6540697523698289, f1_train: 0.8966780785843255\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:13<00:00,  8.24it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 15.95it/s]\n",
      "100%|██████████| 111/111 [00:07<00:00, 15.47it/s]\n",
      "  1%|          | 1/111 [00:00<00:14,  7.60it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 53, f1_test: 0.5979072065353643, f1_train: 0.8376848861974048\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:13<00:00,  8.07it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 15.42it/s]\n",
      "100%|██████████| 111/111 [00:07<00:00, 15.44it/s]\n",
      "  1%|          | 1/111 [00:00<00:14,  7.67it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 54, f1_test: 0.582892471187802, f1_train: 0.8061160909971169\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:13<00:00,  8.10it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 14.75it/s]\n",
      "100%|██████████| 111/111 [00:06<00:00, 16.05it/s]\n",
      "  1%|          | 1/111 [00:00<00:14,  7.57it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 55, f1_test: 0.5926460967201866, f1_train: 0.8180962275848912\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:13<00:00,  8.07it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 15.94it/s]\n",
      "100%|██████████| 111/111 [00:07<00:00, 14.78it/s]\n",
      "  1%|          | 1/111 [00:00<00:14,  7.60it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 56, f1_test: 0.5866476535193236, f1_train: 0.8027100360425875\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:13<00:00,  8.28it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 16.94it/s]\n",
      "100%|██████████| 111/111 [00:07<00:00, 14.82it/s]\n",
      "  1%|          | 1/111 [00:00<00:14,  7.54it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 57, f1_test: 0.5576817707664975, f1_train: 0.7627134282126853\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:13<00:00,  8.14it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 16.07it/s]\n",
      "100%|██████████| 111/111 [00:07<00:00, 14.76it/s]\n",
      "  1%|          | 1/111 [00:00<00:14,  7.59it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 58, f1_test: 0.6403069437722618, f1_train: 0.8869356171278581\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:13<00:00,  8.24it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 16.50it/s]\n",
      "100%|██████████| 111/111 [00:07<00:00, 14.80it/s]\n",
      "  1%|          | 1/111 [00:00<00:14,  7.64it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 59, f1_test: 0.6233325155086638, f1_train: 0.8314687704112131\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:13<00:00,  8.23it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 14.48it/s]\n",
      "100%|██████████| 111/111 [00:07<00:00, 15.82it/s]\n",
      "  1%|          | 1/111 [00:00<00:14,  7.57it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 60, f1_test: 0.6111797245309151, f1_train: 0.8500943930465292\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:14<00:00,  7.67it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 14.94it/s]\n",
      "100%|██████████| 111/111 [00:07<00:00, 15.44it/s]\n",
      "  1%|          | 1/111 [00:00<00:14,  7.40it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 61, f1_test: 0.5841503500434735, f1_train: 0.8152748429040421\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:13<00:00,  8.18it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 16.87it/s]\n",
      "100%|██████████| 111/111 [00:07<00:00, 15.71it/s]\n",
      "  1%|          | 1/111 [00:00<00:14,  7.67it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 62, f1_test: 0.6490465731669248, f1_train: 0.8738844130944925\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:14<00:00,  7.83it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 16.52it/s]\n",
      "100%|██████████| 111/111 [00:07<00:00, 15.29it/s]\n",
      "  1%|          | 1/111 [00:00<00:14,  7.58it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 63, f1_test: 0.5980012113678181, f1_train: 0.819128921017111\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:13<00:00,  8.10it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 15.58it/s]\n",
      "100%|██████████| 111/111 [00:07<00:00, 15.53it/s]\n",
      "  1%|          | 1/111 [00:00<00:14,  7.61it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 64, f1_test: 0.5753841921511882, f1_train: 0.7936820605801598\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:13<00:00,  8.12it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 16.69it/s]\n",
      "100%|██████████| 111/111 [00:07<00:00, 15.76it/s]\n",
      "  1%|          | 1/111 [00:00<00:14,  7.59it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 65, f1_test: 0.5669555213556223, f1_train: 0.7864931656562749\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:14<00:00,  7.74it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 16.60it/s]\n",
      "100%|██████████| 111/111 [00:07<00:00, 15.34it/s]\n",
      "  1%|          | 1/111 [00:00<00:14,  7.55it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 66, f1_test: 0.5758809289694308, f1_train: 0.7894801496741621\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:13<00:00,  8.02it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 16.98it/s]\n",
      "100%|██████████| 111/111 [00:07<00:00, 15.72it/s]\n",
      "  1%|          | 1/111 [00:00<00:14,  7.60it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 67, f1_test: 0.5793752970296472, f1_train: 0.7995635484922041\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:13<00:00,  7.98it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 16.83it/s]\n",
      "100%|██████████| 111/111 [00:06<00:00, 15.89it/s]\n",
      "  1%|          | 1/111 [00:00<00:14,  7.58it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 68, f1_test: 0.6533397293366141, f1_train: 0.8787712033238771\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:13<00:00,  8.19it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 16.79it/s]\n",
      "100%|██████████| 111/111 [00:07<00:00, 15.72it/s]\n",
      "  1%|          | 1/111 [00:00<00:14,  7.60it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 69, f1_test: 0.6143461645590067, f1_train: 0.8316362203853974\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:13<00:00,  7.95it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 15.96it/s]\n",
      "100%|██████████| 111/111 [00:06<00:00, 15.87it/s]\n",
      "  1%|          | 1/111 [00:00<00:14,  7.41it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 70, f1_test: 0.5672506727198753, f1_train: 0.7680698174491756\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:14<00:00,  7.93it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 16.55it/s]\n",
      "100%|██████████| 111/111 [00:07<00:00, 15.33it/s]\n",
      "  1%|          | 1/111 [00:00<00:18,  5.93it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 71, f1_test: 0.6117411214544367, f1_train: 0.8468875587473783\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:15<00:00,  7.34it/s]\n",
      "100%|██████████| 28/28 [00:02<00:00, 12.51it/s]\n",
      "100%|██████████| 111/111 [00:08<00:00, 13.77it/s]\n",
      "  1%|          | 1/111 [00:00<00:15,  6.97it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 72, f1_test: 0.5903563306658821, f1_train: 0.8322547375454714\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:14<00:00,  7.60it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 17.11it/s]\n",
      "100%|██████████| 111/111 [00:07<00:00, 15.19it/s]\n",
      "  1%|          | 1/111 [00:00<00:14,  7.48it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 73, f1_test: 0.5746663070254596, f1_train: 0.7936453581633662\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:13<00:00,  8.37it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 15.35it/s]\n",
      "100%|██████████| 111/111 [00:07<00:00, 15.15it/s]\n",
      "  1%|          | 1/111 [00:00<00:14,  7.56it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 74, f1_test: 0.5964844261354332, f1_train: 0.8198766081960296\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:13<00:00,  8.18it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 17.03it/s]\n",
      "100%|██████████| 111/111 [00:07<00:00, 15.46it/s]\n",
      "  1%|          | 1/111 [00:00<00:14,  7.58it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 75, f1_test: 0.5799678401610108, f1_train: 0.7962146896867508\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:13<00:00,  8.21it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 16.83it/s]\n",
      "100%|██████████| 111/111 [00:06<00:00, 16.15it/s]\n",
      "  1%|          | 1/111 [00:00<00:14,  7.55it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 76, f1_test: 0.5983982955591436, f1_train: 0.8288971403228306\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:14<00:00,  7.72it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 14.00it/s]\n",
      "100%|██████████| 111/111 [00:09<00:00, 11.60it/s]\n",
      "  1%|          | 1/111 [00:00<00:15,  7.24it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 77, f1_test: 0.5920983485153527, f1_train: 0.8340713921309678\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:15<00:00,  7.25it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 15.64it/s]\n",
      "100%|██████████| 111/111 [00:06<00:00, 15.89it/s]\n",
      "  1%|          | 1/111 [00:00<00:14,  7.57it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 78, f1_test: 0.6151718495229238, f1_train: 0.8463174368624474\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:13<00:00,  8.24it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 16.83it/s]\n",
      "100%|██████████| 111/111 [00:07<00:00, 14.80it/s]\n",
      "  1%|          | 1/111 [00:00<00:14,  7.62it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 79, f1_test: 0.5680156569605705, f1_train: 0.8037536913464516\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:13<00:00,  8.16it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 16.03it/s]\n",
      "100%|██████████| 111/111 [00:07<00:00, 14.81it/s]\n",
      "  1%|          | 1/111 [00:00<00:14,  7.65it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 80, f1_test: 0.5953446185682619, f1_train: 0.8253539916686162\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:13<00:00,  8.31it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 15.71it/s]\n",
      "100%|██████████| 111/111 [00:06<00:00, 15.86it/s]\n",
      "  1%|          | 1/111 [00:00<00:14,  7.63it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 81, f1_test: 0.5761642475136113, f1_train: 0.7977725837347271\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:13<00:00,  8.04it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 15.17it/s]\n",
      "100%|██████████| 111/111 [00:08<00:00, 12.57it/s]\n",
      "  1%|          | 1/111 [00:00<00:15,  7.22it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 82, f1_test: 0.5961413184294474, f1_train: 0.8240459726751141\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:14<00:00,  7.55it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 16.87it/s]\n",
      "100%|██████████| 111/111 [00:07<00:00, 15.43it/s]\n",
      "  1%|          | 1/111 [00:00<00:14,  7.62it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 83, f1_test: 0.5741595580947121, f1_train: 0.7943838298842252\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:13<00:00,  8.19it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 16.74it/s]\n",
      "100%|██████████| 111/111 [00:07<00:00, 14.92it/s]\n",
      "  0%|          | 0/111 [00:00<?, ?it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 84, f1_test: 0.5867677109048006, f1_train: 0.8238582879556913\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:13<00:00,  8.09it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 16.59it/s]\n",
      "100%|██████████| 111/111 [00:07<00:00, 15.84it/s]\n",
      "  1%|          | 1/111 [00:00<00:14,  7.57it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 85, f1_test: 0.6101148049296465, f1_train: 0.8392029636262524\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:13<00:00,  8.22it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 16.91it/s]\n",
      "100%|██████████| 111/111 [00:07<00:00, 15.29it/s]\n",
      "  1%|          | 1/111 [00:00<00:14,  7.60it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 86, f1_test: 0.5890405435417683, f1_train: 0.820247113798823\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:13<00:00,  8.30it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 15.72it/s]\n",
      "100%|██████████| 111/111 [00:07<00:00, 13.99it/s]\n",
      "  1%|          | 1/111 [00:00<00:15,  6.93it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 87, f1_test: 0.5980981549755122, f1_train: 0.8356972736466732\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:15<00:00,  7.11it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 14.37it/s]\n",
      "100%|██████████| 111/111 [00:07<00:00, 14.72it/s]\n",
      "  1%|          | 1/111 [00:00<00:14,  7.64it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 88, f1_test: 0.6107625537127437, f1_train: 0.828531595933628\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:13<00:00,  8.24it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 16.81it/s]\n",
      "100%|██████████| 111/111 [00:07<00:00, 14.79it/s]\n",
      "  1%|          | 1/111 [00:00<00:14,  7.62it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 89, f1_test: 0.5973138493476995, f1_train: 0.826230694138632\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:13<00:00,  7.95it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 16.91it/s]\n",
      "100%|██████████| 111/111 [00:07<00:00, 15.17it/s]\n",
      "  1%|          | 1/111 [00:00<00:14,  7.62it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 90, f1_test: 0.6252921004675466, f1_train: 0.8318156346116717\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:13<00:00,  8.21it/s]\n",
      "100%|██████████| 28/28 [00:02<00:00, 12.71it/s]\n",
      "100%|██████████| 111/111 [00:07<00:00, 15.61it/s]\n",
      "  1%|          | 1/111 [00:00<00:14,  7.63it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 91, f1_test: 0.6178930636089086, f1_train: 0.8570748523307519\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:13<00:00,  8.18it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 16.34it/s]\n",
      "100%|██████████| 111/111 [00:07<00:00, 14.75it/s]\n",
      "  1%|          | 1/111 [00:00<00:14,  7.65it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 92, f1_test: 0.5810700157798249, f1_train: 0.8219680630526542\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:15<00:00,  7.00it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 16.31it/s]\n",
      "100%|██████████| 111/111 [00:08<00:00, 13.06it/s]\n",
      "  1%|          | 1/111 [00:00<00:14,  7.63it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 93, f1_test: 0.5858783355916727, f1_train: 0.8020294917334412\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:13<00:00,  8.05it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 14.44it/s]\n",
      "100%|██████████| 111/111 [00:07<00:00, 15.50it/s]\n",
      "  1%|          | 1/111 [00:00<00:14,  7.63it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 94, f1_test: 0.6033179688121204, f1_train: 0.8262599397800517\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:14<00:00,  7.86it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 16.17it/s]\n",
      "100%|██████████| 111/111 [00:07<00:00, 15.56it/s]\n",
      "  1%|          | 1/111 [00:00<00:14,  7.54it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 95, f1_test: 0.6104338015295782, f1_train: 0.8176572660797187\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:13<00:00,  8.23it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 16.49it/s]\n",
      "100%|██████████| 111/111 [00:07<00:00, 15.52it/s]\n",
      "  1%|          | 1/111 [00:00<00:14,  7.61it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 96, f1_test: 0.6005834219551153, f1_train: 0.8367462670515983\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:13<00:00,  8.11it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 16.37it/s]\n",
      "100%|██████████| 111/111 [00:07<00:00, 15.12it/s]\n",
      "  1%|          | 1/111 [00:00<00:20,  5.49it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 97, f1_test: 0.5828030508041074, f1_train: 0.8271996039726968\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:14<00:00,  7.91it/s]\n",
      "100%|██████████| 28/28 [00:01<00:00, 14.47it/s]\n",
      "100%|██████████| 111/111 [00:07<00:00, 14.92it/s]\n",
      "  1%|          | 1/111 [00:00<00:14,  7.56it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 98, f1_test: 0.6181327235393923, f1_train: 0.8340865443475837\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 111/111 [00:13<00:00,  7.98it/s]\n",
      "100%|██████████| 28/28 [00:02<00:00, 13.00it/s]\n",
      "100%|██████████| 111/111 [00:07<00:00, 15.50it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch: 99, f1_test: 0.6140860930295599, f1_train: 0.8440663933052893\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    }
   ],
   "source": [
    "n_epoch = 100\n",
    "best_f1 = 0\n",
    "for epoch in range(n_epoch):\n",
    "    model.train()\n",
    "    for wavs, labs in tqdm(train_loader):\n",
    "        optimizer.zero_grad()\n",
    "        wavs, labs = wavs.cuda(), labs.cuda()\n",
    "        outputs = model(wavs)\n",
    "        loss = criterion(outputs, labs)\n",
    "        loss.backward()\n",
    "        optimizer.step()\n",
    "#     if epoch % 10 == 0:\n",
    "    f1 = eval_model(model, val_loader)\n",
    "    f1_train = eval_model(model, train_loader)\n",
    "    print(f'epoch: {epoch}, f1_test: {f1}, f1_train: {f1_train}')\n",
    "    if f1 > best_f1:\n",
    "        best_f1 = f1\n",
    "        torch.save(model.state_dict(), '../baseline_fulldiv.pt')\n",
    "        \n",
    "    lr = lr * 0.95\n",
    "    for param_group in optimizer.param_groups:\n",
    "        param_group['lr'] = lr"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 82,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "  0%|          | 0/93 [00:00<?, ?it/s]/home/ilin-a@ad.speechpro.com/miniconda3/envs/itmo-baseline/lib/python3.7/site-packages/ipykernel_launcher.py:60: UserWarning: Implicit dimension choice for softmax has been deprecated. Change the call to include dim=X as an argument.\n",
      "  2%|▏         | 2/93 [00:00<00:05, 15.58it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Loaded pretrained weights for efficientnet-b0\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 93/93 [00:06<00:00, 15.36it/s]\n"
     ]
    }
   ],
   "source": [
    "# make a model\n",
    "model_name = 'baseline_fulldiv.pt'\n",
    "model = BaseLineModel().cuda()\n",
    "model.load_state_dict(torch.load(os.path.join('..', model_name)))\n",
    "model.eval()\n",
    "forecast = []\n",
    "with torch.no_grad():\n",
    "    for wavs in tqdm(test_loader):\n",
    "        wavs = wavs.cuda()\n",
    "        outputs = model.inference(wavs)\n",
    "        outputs = outputs.detach().cpu().numpy().argmax(axis=1)\n",
    "        forecast.append(outputs)\n",
    "forecast = [x for sublist in forecast for x in sublist]\n",
    "decoder = {classes_dict[cl]:cl for cl in classes_dict}\n",
    "forecast = pd.Series(forecast).map(decoder)\n",
    "df_test['label'] = forecast\n",
    "df_test.to_csv(f'{model_name}.csv', index=None)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
