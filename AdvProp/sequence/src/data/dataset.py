from torch.utils.data import Dataset
import torch
import random
import numpy as np
from rdkit import Chem

ALPHABET = "0" + "Cc1(O)=N23S4nF.l[-]sa+oKH#BrgP56ZIb78iLAteGd9" + "\\/@%PuYmpfMhW*VUETDy~R" + "<>?"

class TargetSet(Dataset):
    def __init__(self, smile_list, label_list, use_aug=False, use_cls_token=False, seq_len=None):
        # vocabulary
        alphabet = ALPHABET
        self.char_indices = dict((c, i) for i, c in enumerate(alphabet))
        self.vocab_size = len(self.char_indices)
        print("dictionary size: " + str(len(self.char_indices)))

        if label_list is not None:
            self.labels = np.array(label_list)
        else:
            self.labels = np.zeros([len(smile_list),1])
        self.mol_list, smile_len_max = self._read_mol(smile_list)
        self.corpus_lines = len(self.mol_list)

        if seq_len is None:
            self.seq_len = smile_len_max + 80
        else:
            self.seq_len = seq_len

        self.use_aug = use_aug
        self.use_cls_token = use_cls_token

    def _read_mol(self, smile_list):
        mol_list = []
        smile_len_max = 0
        for smile_str in smile_list:
            smile_len_max = max(len(smile_str), smile_len_max)
            m = Chem.MolFromSmiles(smile_str)
            mol_list.append(m)
            
        return mol_list, smile_len_max

    def _mol_to_smile(self, mol):
        if not self.use_aug:
            return Chem.MolToSmiles(mol, canonical=True)
        else:
            new_atom_order = list(range(mol.GetNumAtoms()))
            random.shuffle(new_atom_order)
            random_mol = Chem.RenumberAtoms(mol, newOrder=new_atom_order)
            return Chem.MolToSmiles(random_mol, canonical=False, isomericSmiles=True)

    def _smile_to_seq(self, smile):
        seq = [self.char_indices[atom] for atom in smile]
        return seq

    def _sample(self, idx):
        mol, label = self.mol_list[idx], self.labels[idx]
        smile = self._mol_to_smile(mol)
        seq = self._smile_to_seq(smile)
        mask = [x is not None for x in label]
        label = [x if x is not None else 0 for x in label]
        return seq, label, mask

    def _sample_can(self, idx):
        mol = self.mol_list[idx]
        smile = Chem.MolToSmiles(mol, canonical=True)
        seq = [self.char_indices[atom] for atom in smile]
        return seq

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, idx):
        mol_seq, mol_label, mask = self._sample(idx)
        length = len(mol_seq)

        if length >= self.seq_len:
            mol_seq = self._sample_can(idx)
            length = len(mol_seq)
            if length >= self.seq_len:
                while length >= self.seq_len:
                    idx = np.random.randint(0, self.corpus_lines)
                    mol_seq, mol_label, mask = self._sample(idx)
                    length = len(mol_seq)

        if self.use_cls_token:
            mol_seq = [self.vocab_size-3] + mol_seq
            length += 1

        padding = [0 for _ in range(self.seq_len - length)]
        mol_seq.extend(padding)

        seq_as_tensor = torch.tensor(mol_seq).long()
        length_as_tensor = torch.tensor(length).long()
        label_as_tensor = torch.tensor(mol_label).float()
        mask_as_tensor = torch.tensor(mask).float()

        output = {
            "idx": idx,
            "seq_input": seq_as_tensor,
            "length": length_as_tensor,
            "label": label_as_tensor,
            "mask": mask_as_tensor
        }

        return output
    
    def get_imblance_ratio(self):
        return ((self.labels==0).sum())/((self.labels==1).sum())



class PretrainSet(Dataset):
    def __init__(self, smile_list, task='mask_con', use_aug=False, use_cls_token=False, seq_len=None):
        # vocabulary
        alphabet = ALPHABET
        self.char_indices = dict((c, i) for i, c in enumerate(alphabet))
        self.vocab_size = len(self.char_indices)
        print("dictionary size: " + str(len(self.char_indices)))

        self.mol_list, smile_len_max = self._read_mol(smile_list)
        self.task = task
        self.corpus_lines = len(self.mol_list)

        if seq_len is None:
            self.seq_len = smile_len_max + 80
        else:
            self.seq_len = seq_len

        self.use_aug = use_aug
        self.use_cls_token = use_cls_token
    
    def _read_mol(self, smile_list):
        mol_list = []
        smile_len_max = 0
        for smile_str in smile_list:
            smile_len_max = max(len(smile_str), smile_len_max)
            m = Chem.MolFromSmiles(smile_str)
            mol_list.append(m)
        return mol_list, smile_len_max

    def _mol_to_smile(self, mol):
        if not self.use_aug:
            return Chem.MolToSmiles(mol, canonical=True)
        else:
            new_atom_order = list(range(mol.GetNumAtoms()))
            random.shuffle(new_atom_order)
            random_mol = Chem.RenumberAtoms(mol, newOrder=new_atom_order)
            return Chem.MolToSmiles(random_mol, canonical=False, isomericSmiles=True)

    def _smile_to_seq(self, smile):
        seq = [self.char_indices[atom] for atom in smile]
        return seq

    def _sample(self, idx):
        mol = self.mol_list[idx]
        smile = self._mol_to_smile(mol)
        seq = self._smile_to_seq(smile)
        return seq

    def __len__(self):
        return self.corpus_lines  # 2,000,000 too large? not very large

    def _random_smile(self, idx):
        is_same = random.randint(0,1)
        if is_same:
            return self._sample(idx), is_same
        else:
            random_idx = random.randint(0, self.corpus_lines - 2)
            random_idx = random_idx if random_idx < idx else random_idx + 1
            return self._sample(random_idx), is_same

    def __getitem__(self, idx):
        if self.task == 'mask_pred':
            mol_seq = self._sample(idx)
            length = len(mol_seq)
            while length >= self.seq_len:
                idx = np.random.randint(0, self.corpus_lines)
                mol_seq = self._sample(idx)
                length = len(mol_seq)
            
            masked_mol_seq, _, atom_labels = self.random_atom(mol_seq)

            if self.use_cls_token:
                masked_mol_seq = [self.vocab_size-3] + masked_mol_seq
                atom_labels = [-1] + atom_labels
                length += 1

            padding = [0 for _ in range(self.seq_len - length)]
            masked_mol_seq.extend(padding)
            pad_labels = [-1 for _ in range(self.seq_len - length)]
            atom_labels.extend(pad_labels)

            seq_as_tensor = torch.tensor(masked_mol_seq).long()
            atom_labels_as_tensor = torch.tensor(atom_labels).long()
            output = {"seq_mask": seq_as_tensor,
                    "atom_labels": atom_labels_as_tensor
                    }

        elif self.task == 'mask_con':
            mol_seq = self._sample(idx)
            length = len(mol_seq)
            while length >= self.seq_len:
                idx = np.random.randint(0, self.corpus_lines)
                mol_seq = self._sample(idx)
                length = len(mol_seq)

            masked_mol_seq, labels, _ = self.random_atom(mol_seq)

            if self.use_cls_token:
                mol_seq = [self.vocab_size-3] + mol_seq
                masked_mol_seq = [self.vocab_size-3] + masked_mol_seq
                labels = [-1] + labels
                length += 1

            padding = [0 for _ in range(self.seq_len - length)]
            mol_seq.extend(padding)
            masked_mol_seq.extend(padding)
            pad_labels = [-1 for _ in range(self.seq_len - length)]
            labels.extend(pad_labels)

            seq_as_tensor1, seq_as_tensor2, labels_as_tensor = torch.tensor(mol_seq).long(), torch.tensor(masked_mol_seq).long(), torch.tensor(labels).long()

            output = {"seq": seq_as_tensor1,
                    "seq_mask": seq_as_tensor2,
                    "labels": labels_as_tensor
                    }

        else:
            raise ValueError('not supported pretrain task!')

        return output

    def random_atom(self, molecule):  # molecule: [3,1,2,4,3,...]
        masked_m = molecule.copy()
        labels = []
        atom_labels = []
        no_masked = True

        for i, atom_idx in enumerate(molecule):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    masked_m[i] = self.vocab_size - 1
                    if self.use_cls_token:
                        labels.append(i+1)
                    else:
                        labels.append(i)
                    atom_labels.append(atom_idx)
                    no_masked = False

                # 10% randomly change token to random token
                elif prob < 0.9:
                    masked_m[i] = random.randrange(1, self.vocab_size-4)
                    if self.use_cls_token:
                        labels.append(i+1)
                    else:
                        labels.append(i)
                    atom_labels.append(atom_idx)
                    no_masked = False

                # 10% randomly change token to current token
                else:
                    masked_m[i] = atom_idx
                    atom_labels.append(-1)
                    labels.append(-1)

            else:
                masked_m[i] = atom_idx
                atom_labels.append(-1)
                labels.append(-1)
        
        if  no_masked and len(labels) > 1:
            mask_idx = random.randint(0, i)
            atom_idx = masked_m[mask_idx]
            masked_m[mask_idx] = self.vocab_size - 1
            if self.use_cls_token:
                labels[mask_idx] = mask_idx + 1
            else:
                labels[mask_idx] = mask_idx
            atom_labels[mask_idx] = atom_idx
            
        return masked_m, labels, atom_labels