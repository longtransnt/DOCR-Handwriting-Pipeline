from TextRecognition.vietocr.TextRecognition import TextRecognition
from TextRecognition.vietocr.vietocr.tool.utils import compute_accuracy
import os

vgg19_transformer = TextRecognition()
if(vgg19_transformer):
    print(" ✔ Text Recognition   -   VGG19-Transormer model loaded")
else:
    raise ValueError(
        '❌ Text Detection - VGG19-Transormer model failed to load')

prefix = "/mnt/d/DOCR/OUCRU-Handwriting-Pipeline/Output/Adaptive"

test_list_file = "/mnt/d/DOCR/OUCRU-Handwriting-Pipeline/TextRecognition/vietocr/Sorted/test_resorted_adaptive"
f = open(test_list_file + ".txt", "r", encoding="utf8")

preds = []
exps = []
for line in f:
    # print(line, end = "")

    splits = line.split("\t")
    file_path = splits[0]
    file_path = file_path[1:]

    img_path = prefix + file_path
    if (os.path.exists(img_path) is not True):
        continue
    expected = " ".join(splits[1:])
    print("=" * 20)
    prediction = vgg19_transformer.infer(img_path, ngram=True)
    print(" - Final prediction: ", prediction)
    print(" - Expected prediction: ", expected)

    preds.append(prediction)
    exps.append(expected)
full_sequence = compute_accuracy(
    predictions=preds, expected=exps, mode="full_sequence")
per_char = compute_accuracy(
    predictions=preds, expected=exps, mode="per_char")
cer = compute_accuracy(
    predictions=preds, expected=exps, mode="cer")
wer = compute_accuracy(
    predictions=preds, expected=exps, mode="per_word")
info = 'acc full seq: {:.4f} - acc per char: {:.4f} - WER {:.4f} - CER {:.4f}'.format(
    full_sequence, per_char, wer, cer)
print(info)
