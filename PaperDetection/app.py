import json

from PIL import Image
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State

from MaskCRNN import MaskCRNN
from TextDetector import TextDetector
from TextPredictor import TextPredictor
from helpers import parse_image
from text_classifier import TextClassifier
from tuan_utils import sort_rec, calculate_angle_rectangles, is_valid_angle, is_confirm_by_shifted, get_max_x_rect

intro_text = """
Billing prototyping
"""

mask = MaskCRNN()
craft = TextDetector()
transformer_ocr = TextPredictor()
app = dash.Dash(name=__name__)
app.title = 'Bill Extractor'

server = app.server

app.css.config.serve_locally = False
app.config.suppress_callback_exceptions = True

header = html.Div(
    id="app-header",
    children=[
        html.Img(src=app.get_asset_url("logo.png"), className="logo"),
        "Bill Detection 0.99",
    ],
)

app.layout = html.Div(
    children=[
        header,
        html.Br(),
        html.Details(
            id="intro-text",
            children=[html.Summary(html.B("About This App")), dcc.Markdown(intro_text)],
        ),
        # html.Div(html.Div(id="intro-text", children=dcc.Markdown(intro_text),),),
        html.Div(
            id="app-body",
            children=[
                html.Div(
                    style={"width": "75vw"},
                    children=[
                        dcc.Tabs([
                            dcc.Tab(label='Bounding Box Detection', id='bound-tab', children=[
                            ]),
                            dcc.Tab(label='Text Detection', id='text-detect-tab', children=[
                            ]),
                        ]),
                    ],
                ),
                html.Div(
                    id="control-card",
                    style={"margin-left": "15px"},
                    children=[
                        html.Span(
                            className="control-label", children="Upload an Image"
                        ),
                        dcc.Upload(
                            id="img-upload",
                            className="upload-component",
                            children=html.Div(
                                ["Drag and Drop or ", html.A("Select Files")]
                            ),
                        ),
                        html.Div(id="output-img-upload"),
                    ],
                ),
            ],
        ),
    ]
)


@app.callback(
    [Output("output-img-upload", "children"),
     Output("bound-tab", "children"),
     Output("text-detect-tab", "children")],
    [Input("img-upload", "contents")],
    [State("img-upload", "filename"), State("img-upload", "last_modified")],
)
def display_uploaded_img(contents, fname, date):
    if contents is not None:
        original_img = parse_image(contents, fname, date)

        predicted_bb, cropped_img = mask.predict(im=original_img)
        boxes, polys, ret_score_text, poly_cropped_img = craft.detect(cropped_img)
        console.log(boxes)

        # Have to convert all to PIL image for display
        # pil_image = Image.fromarray(original_img)  # Original
        predicted_pil_image = Image.fromarray(predicted_bb)  # Predicted
        pil_poly_cropped_img = Image.fromarray(poly_cropped_img)  # Cropped and detected

        prediction_texts, rectangles, probabilities = transformer_ocr.predict_bill(cropped_img, polys)
        print(prediction_texts)
        rectangles = map(sort_rec, rectangles)
        current_res_list = list(zip(prediction_texts, rectangles, probabilities))
        current_res_list = sorted(current_res_list, key=lambda rl: min(rl[1], key=lambda tx: tx[0])[1], reverse=False)

        text = ""
        text_for_bert = ""
        current_line = []
        all_line = []

        for j, image_result in enumerate(current_res_list):
            res_text, rect, p = image_result
            if rect[0][1] > rect[3][1]:
                rect[3][1] = rect[0][1]
            else:
                rect[0][1] = rect[3][1]
            if rect[1][1] > rect[2][1]:
                rect[2][1] = rect[1][1]
            else:
                rect[1][1] = rect[2][1]

            if j > 0:
                prev_rect = list(current_res_list)[j - 1][1]
                r = calculate_angle_rectangles(
                    prev_rect, rect)
                if r:
                    intersect_angle_l, intersect_angle_r, intersect_angle_u, intersect_angle_d, \
                    intersect_angle_l_2, intersect_angle_r_2, shifted_intersect_angle_l_3, shifted_intersect_angle_r_3, y_overlapped_before_resize_percentage, intersect_result = r

                    if is_valid_angle(intersect_angle_l) and is_valid_angle(intersect_angle_r) and \
                            is_valid_angle(intersect_angle_l_2) and is_valid_angle(intersect_angle_r_2):
                        overlapped_previous = False

                        current_range = set(range(rect[0][0].astype(int), rect[3][0].astype(int)))
                        current_range_y = set(range(rect[0][1].astype(int), rect[1][1].astype(int)))
                        for text_block in current_line:
                            range1 = set(range(text_block[1][0][0].astype(int), text_block[1][3][0].astype(int)))
                            range1_y = set(range(text_block[1][0][1].astype(int), text_block[1][1][1].astype(int)))
                            overlapped_percentage = len(current_range.intersection(range1)) / min(len(range1),
                                                                                                  len(current_range))
                            y_overlapped_percentage = len(current_range_y.intersection(range1_y)) / min(len(range1_y),
                                                                                                        len(
                                                                                                            current_range_y))
                            if overlapped_percentage > 0.2:
                                overlapped_previous = True
                                break

                        if overlapped_percentage < 0.2 and is_confirm_by_shifted(
                                abs(shifted_intersect_angle_l_3 - intersect_angle_l_2)):
                            current_line.append((res_text, rect))
                        else:
                            overlapped_previous = False
                            if y_overlapped_before_resize_percentage > 0.6:
                                current_line.append((res_text, rect))
                            else:
                                all_line.append(current_line)
                                current_line = [(res_text, rect)]

                        if overlapped_previous:
                            all_line.append(current_line)
                            current_line = [(res_text, rect)]
                            break
                    else:
                        if y_overlapped_before_resize_percentage > 0.6:
                            current_line.append((res_text, rect))
                        else:
                            all_line.append(current_line)
                            current_line = [(res_text, rect)]
            else:
                current_line = [(res_text, rect)]
        all_line.append(current_line)

        for line in all_line:
            # print(line)
            line = sorted(line, key=lambda sy: get_max_x_rect(sy))
            line_text_list = list(map(lambda l: l[0], line))
            line_text = " ".join(line_text_list)
            text += f"{line_text}\n<br>"
            text_for_bert += "|||".join(line_text_list) + '\n'

        device = 'cuda'
        # device = 'cpu'  # cpu for the impossible :)
        text_classifier = TextClassifier(device)
        ner_result = text_classifier.filter(text_for_bert)
        # print(ner_result)

        children = [
            dcc.Tabs([
                dcc.Tab(label='Text', children=[
                    html.Iframe(srcDoc=text, style={'height': '60vh', 'width': '19.8vw'}),
                ]),
                dcc.Tab(label='NER', children=[
                    html.Iframe(
                        srcDoc=json.dumps(ner_result, ensure_ascii=False, sort_keys=True, indent=0),
                        style={'height': '60vh', 'width': '19.8vw'}),
                ]),
            ]),
        ]
        children2 = [
            html.Img(className="image", src=predicted_pil_image),
        ]
        children3 = [
            html.Img(className="image", src=pil_poly_cropped_img),
        ]
        return children, children2, children3
    return None, None, None


if __name__ == "__main__":
    app.run_server(debug=False)
