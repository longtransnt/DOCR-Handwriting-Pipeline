import re
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from unidecode import unidecode
from bert.dataset import BillCategories
from bert.model import PhoBertFineTune

label_to_ix = {'SELLER': 0,
               'ADDRESS': 1,
               'TIMESTAMP': 2,
               'TOTAL_COST': 3,
               'TOTAL_TOTAL_COST': 4}


def is_price_tag(s):
    return len(re.findall(r"(.?ND)? ?\d+[,. ]?\d+( ?)\w{1,3}.+", s)) > 0


def is_valid_address(line):
    """
    rau cau long hai
    :return:
    is rau cau long hai?
    """
    if count_consecutive_digits(line) >= 5:
        return False
    if len(re.findall(r"^\d+[\.|\,|\ ]?\d+$", line)) > 0:
        return False
    return True


def count_consecutive_digits(address):
    max_consecutive_digits_count = 0
    consecutive_digits_count = 0
    prev_isdigit = False
    for c in address:
        if not prev_isdigit:
            consecutive_digits_count = 0
        if c.isdigit():
            prev_isdigit = True
            consecutive_digits_count += 1
            if consecutive_digits_count > max_consecutive_digits_count:
                max_consecutive_digits_count = consecutive_digits_count
        else:
            prev_isdigit = False
    return max_consecutive_digits_count


# %%
def get_key(val):
    for key, value in label_to_ix.items():
        if val == value:
            return key
    return "key doesn't exist"


class TextClassifier:
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
    model = None
    device = 'cuda'

    def __init__(self, device):
        self.device = device
        self.model = PhoBertFineTune()
        self.model.to(device)
        self.model.load_state_dict(
            torch.load("./weights/phobert_ft.pth", map_location=device))

    def predict(self, pretext):
        warm_up = pd.DataFrame(columns=['text', 'label'])
        warm_up = warm_up.append({'text': pretext, 'label': 'ADDRESS'}, ignore_index=True)
        warm_up['ENCODE_CAT'] = warm_up['label'].apply(lambda x: label_to_ix[x])
        warm_up = warm_up.reset_index(drop=True)
        warm_up_set = BillCategories(warm_up, self.tokenizer, 258)
        train_params = {'batch_size': 1,
                        'shuffle': True,
                        'num_workers': 0
                        }
        testing_loader = DataLoader(warm_up_set, **train_params)
        self.model.eval()
        with torch.no_grad():
            for _, data in enumerate(testing_loader, 0):
                ids = data['ids'].to(self.device, dtype=torch.long)
                mask = data['mask'].to(self.device, dtype=torch.long)
                outputs = self.model(ids, mask)
                big_val, big_idx = torch.max(outputs.data, dim=1)
                sm = torch.nn.Softmax(dim=1)
                probs = sm(outputs).detach().cpu().numpy()
                max_confidence = probs[0][big_idx]
                return big_idx, max_confidence
        return None, None

    def filter(self, all_text):
        address_text = ""
        timestamp_text = ""
        total_text = " "
        sellers = []

        has_total = False
        lines = all_text.split("\n")

        for line_index, line in enumerate(lines):
            for tok in line.split("|||"):
                big_idx, max_confidence = self.predict(tok)
                no_tonal_tok = unidecode(tok).lower()
                # ADDRESS
                if max_confidence > 0.97 and big_idx == 1:
                    if line_index > 0:
                        if not is_valid_address(lines[line_index - 1]):
                            continue

                    if count_consecutive_digits(no_tonal_tok) >= 5:
                        continue
                    if len(no_tonal_tok) < 10:
                        continue
                    if sum(c.isdigit() for c in tok) / len(tok) > 0.5:
                        continue
                    if line_index < 5:
                        if address_text != "":
                            address_text += '|||'
                        address_text += f"{tok}"
                        print(
                            f"{get_key(big_idx).upper()}: {tok}, LineNo: {line_index}, Confidence: {max_confidence}")
                    if line_index >= 5 and max_confidence > 0.9:
                        if line_index < len(lines) - 1:
                            if not is_valid_address(lines[line_index + 1]):
                                continue
                        if not len(re.findall(r'(\.|,)', no_tonal_tok)) > 0:
                            continue
                        if line_index > 10:
                            continue
                        if address_text != "":
                            address_text += '|||'
                        address_text += f"{tok}"
                        print(
                            f"{get_key(big_idx).upper()}: {tok}, LineNo: {line_index}, Confidence: {max_confidence}")
                # DATE
                if big_idx == 2 and max_confidence > 0.97:
                    if 'diem' not in no_tonal_tok and 'hang' not in no_tonal_tok and 'gio lam viec' not in no_tonal_tok \
                            and 'quay' not in no_tonal_tok and 'so gd treo' not in no_tonal_tok:
                        realloc_index = no_tonal_tok.find("ngay") + max(no_tonal_tok.find("thoi gian"), 0) + max(
                            no_tonal_tok.find("gio"), 0)
                        if realloc_index > 0:
                            tok = tok[realloc_index:]
                        else:
                            if sum(c.isdigit() for c in tok) < 3:
                                continue
                            if count_consecutive_digits(no_tonal_tok) >= 5:
                                continue
                            if len(re.findall(r'\d+[.|:|,]\d{3,}', no_tonal_tok)):
                                continue
                        if timestamp_text != "":
                            timestamp_text += '|||'
                        timestamp_text += f"{tok}"
                        print(
                            f"Text: {tok}, LineNo: {line_index}, Class: {get_key(big_idx)}, Confidence: {max_confidence}")
                # SELLER
                if line_index < 5 and max_confidence > 0.8 and big_idx == 0:
                    if len(no_tonal_tok) < 5:
                        continue
                    sellers.append((tok, max_confidence))
                    print(
                        f"SELLER: {tok}, LineNo: {line_index}, Class: {get_key(big_idx)}, Confidence: {max_confidence}")
                if big_idx == 3:
                    if 'tong' in no_tonal_tok or 'tien' in no_tonal_tok:
                        if re.match(r"^(tong).+(oan):?$", no_tonal_tok) or re.match(r"^(tong)\ (tien)$",
                                                                                    no_tonal_tok) \
                                or re.match(r"^(cong).+(hang)$", no_tonal_tok):
                            total_text = line
                            has_total = True
                        else:
                            if (re.match(r"^(tong).+(cong).+$", no_tonal_tok) or
                                re.match(r"^(cong)$", no_tonal_tok) or
                                re.match(r"^(tong)$", no_tonal_tok)) \
                                    and not has_total:
                                total_text = line
                                has_total = True

        print(f"TOTAL: {total_text}, LineNo: {line_index}, Confidence: {max_confidence}")

        if len(sellers) > 0:
            seller = max(sellers, key=lambda x: x[1])
        else:
            seller = '', ''
        print(f"Max Prob Seller is: {seller}")
        result = {
            'SELLER': seller[0],
            'ADDRESS': address_text.split("|||"),
            'TOTAL': total_text.split("|||"),
            'TIMESTAMP': timestamp_text.split("|||")
        }
        # concat_text = f"{seller[0]}|||{address_text}|||{total_text}"
        return result
