"""Build finalreport.pdf from the project results."""
import os
import csv
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.colors import HexColor, white
from reportlab.lib.units import cm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Table,
    TableStyle, Image as RLImage, PageBreak, CondPageBreak, KeepTogether,
)
from reportlab.platypus.tableofcontents import TableOfContents

W, H = A4
RESULTS = "results"
OUT_PDF = "finalreport.pdf"
CONTENT_W = W - 4 * cm

C_BLUE = HexColor("#1F4E79")
C_TEAL = HexColor("#1B6B5B")
C_PURPLE = HexColor("#5B2A86")
C_DARK = HexColor("#212121")
C_GREY = HexColor("#8A8A8A")
C_LIGHT = HexColor("#DCE6F1")
C_ROW = HexColor("#F2F6FB")
C_GREEN = HexColor("#2E7D32")
C_TERMBG = HexColor("#1E1E1E")
C_CODEBG = HexColor("#ECEFF1")


def pstyle(name, **kw):
    return ParagraphStyle(name, **kw)


TITLE = pstyle("Title", fontName="Helvetica-Bold", fontSize=26, textColor=C_BLUE,
               leading=32, alignment=TA_CENTER, spaceAfter=6)
SUBTITLE = pstyle("Subtitle", fontName="Helvetica-Bold", fontSize=14, textColor=C_BLUE,
                  leading=20, alignment=TA_CENTER, spaceAfter=4)
COVER = pstyle("Cover", fontName="Helvetica", fontSize=11, textColor=C_DARK,
               leading=18, alignment=TA_CENTER)
AUTHOR = pstyle("Author", fontName="Helvetica-Bold", fontSize=13, textColor=C_DARK,
                leading=18, alignment=TA_CENTER, spaceBefore=10, spaceAfter=4)

H1BLUE = pstyle("PartHeaderBlue", fontName="Helvetica-Bold", fontSize=14, textColor=white,
                backColor=C_BLUE, leading=20, borderPadding=(7, 10, 9, 10),
                spaceBefore=14, spaceAfter=12)
H1TEAL = pstyle("PartHeaderTeal", fontName="Helvetica-Bold", fontSize=14, textColor=white,
                backColor=C_TEAL, leading=20, borderPadding=(7, 10, 9, 10),
                spaceBefore=14, spaceAfter=12)
H1PURPLE = pstyle("PartHeaderPurple", fontName="Helvetica-Bold", fontSize=14, textColor=white,
                  backColor=C_PURPLE, leading=20, borderPadding=(7, 10, 9, 10),
                  spaceBefore=14, spaceAfter=12)
H1DARK = pstyle("PartHeaderDark", fontName="Helvetica-Bold", fontSize=14, textColor=white,
                backColor=C_DARK, leading=20, borderPadding=(7, 10, 9, 10),
                spaceBefore=14, spaceAfter=12)

H1PLAIN = pstyle("SectionPlain", fontName="Helvetica-Bold", fontSize=15, textColor=C_BLUE,
                 leading=20, spaceBefore=12, spaceAfter=8)
H2 = pstyle("SubHeader", fontName="Helvetica-Bold", fontSize=12, textColor=C_BLUE,
            leading=16, spaceBefore=10, spaceAfter=4)
H3 = pstyle("MinorHeader", fontName="Helvetica-Bold", fontSize=10.5, textColor=C_DARK,
            leading=15, spaceBefore=7, spaceAfter=3)
QUESTION = pstyle("Question", fontName="Helvetica-Oblique", fontSize=10.5, textColor=C_BLUE,
                  leading=15, leftIndent=10, spaceBefore=4, spaceAfter=5)
BODY = pstyle("Body", fontName="Helvetica", fontSize=9.7, textColor=C_DARK,
              leading=14, alignment=TA_JUSTIFY, spaceBefore=3, spaceAfter=3)
BULLET = pstyle("Bullet", fontName="Helvetica", fontSize=9.7, textColor=C_DARK,
                leading=14, leftIndent=16, bulletIndent=4, spaceBefore=1, spaceAfter=1)
CAPTION = pstyle("FigCaption", fontName="Helvetica-Oblique", fontSize=8.3, textColor=C_GREY,
                 leading=11, alignment=TA_CENTER, spaceBefore=3, spaceAfter=8)
TABCAP = pstyle("TableCaption", fontName="Helvetica-Bold", fontSize=8.6, textColor=C_DARK,
                leading=12, spaceBefore=4, spaceAfter=3)
CODE = pstyle("Code", fontName="Courier", fontSize=8.2, textColor=HexColor("#263238"),
              backColor=C_CODEBG, leading=11.5, leftIndent=8, rightIndent=8,
              borderPadding=(6, 6, 6, 6), spaceBefore=3, spaceAfter=6)
TERM = pstyle("Term", fontName="Courier", fontSize=8, textColor=HexColor("#E6E6E6"),
              backColor=C_TERMBG, leading=11, leftIndent=8, rightIndent=8,
              borderPadding=(6, 6, 6, 6), spaceBefore=3, spaceAfter=6)
CODELAB = pstyle("CodeLabel", fontName="Helvetica-Bold", fontSize=8, textColor=C_GREEN,
                 leading=11, spaceBefore=4, spaceAfter=1)


def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;")


def mono(text):
    t = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    t = t.replace("\n", "<br/>").replace(" ", "&nbsp;")
    return t


def code(text, label=None):
    items = []
    if label:
        items.append(Paragraph(esc(label), CODELAB))
    items.append(Paragraph(mono(text), CODE))
    return KeepTogether(items)


def term(text, label=None):
    items = []
    if label:
        items.append(Paragraph(esc(label), CODELAB))
    items.append(Paragraph(mono(text), TERM))
    return KeepTogether(items)


def para(text):
    return Paragraph(esc(text), BODY)


def bullet(text):
    return Paragraph(esc(text), BULLET, bulletText="•")


def h2(text):
    return Paragraph(esc(text), H2)


def h3(text):
    return Paragraph(esc(text), H3)


def question(text):
    return Paragraph(esc(text), QUESTION)


def tcap(text):
    return Paragraph(esc(text), TABCAP)


def part_header(text, style):
    return Paragraph(esc(text), style)


def info_table(rows):
    col_w = [5.6 * cm, CONTENT_W - 5.6 * cm]
    data = [[Paragraph("<b>%s</b>" % esc(k), BODY), Paragraph(esc(v), BODY)] for k, v in rows]
    tbl = Table(data, colWidths=col_w)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), C_LIGHT),
        ("TEXTCOLOR", (0, 0), (0, -1), C_BLUE),
        ("GRID", (0, 0), (-1, -1), 0.4, C_GREY),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    return tbl


def result_table(headers, rows, widths=None):
    data = [[Paragraph("<b>%s</b>" % esc(h), BODY) for h in headers]]
    for row in rows:
        data.append([Paragraph(esc(str(c)), BODY) for c in row])
    if widths is None:
        widths = [CONTENT_W / len(headers)] * len(headers)
    tbl = Table(data, colWidths=widths)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), C_BLUE),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("GRID", (0, 0), (-1, -1), 0.4, C_GREY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, C_ROW]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    return tbl


_FIG = {"n": 0}


def figure(fname, caption, width=15.5 * cm, max_h=10.5 * cm):
    _FIG["n"] += 1
    label = "Figure %d. %s" % (_FIG["n"], caption)
    path = os.path.join(RESULTS, fname)
    if not os.path.exists(path):
        return Paragraph(esc("[missing figure: %s]" % fname), CAPTION)
    iw, ih = ImageReader(path).getSize()
    w = width
    h = w * ih / iw
    if h > max_h:
        h = max_h
        w = h * iw / ih
    img = RLImage(path, width=w, height=h)
    img.hAlign = "CENTER"
    return [CondPageBreak(h + 1.4 * cm), img, Paragraph(esc(label), CAPTION)]


class ReportDoc(BaseDocTemplate):
    def afterFlowable(self, flowable):
        if not isinstance(flowable, Paragraph):
            return
        name = flowable.style.name
        text = flowable.getPlainText()
        if name.startswith("PartHeader"):
            self.notify("TOCEntry", (0, text, self.page))
        elif name == "SubHeader":
            self.notify("TOCEntry", (1, text, self.page))
        elif name == "FigCaption" and text.startswith("Figure"):
            self.notify("FIGEntry", (0, text, self.page))


class FigureList(TableOfContents):
    def notify(self, kind, stuff):
        if kind == "FIGEntry":
            self.addEntry(*stuff)


def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(C_GREY)
    if doc.page > 1:
        canvas.drawCentredString(W / 2.0, 1.2 * cm, str(doc.page))
    canvas.restoreState()


def sp(n=1):
    return Spacer(1, n * 0.32 * cm)


def read_csv(fname):
    rows = []
    path = os.path.join(RESULTS, fname)
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def build():
    story = []

    # Cover
    story += [
        sp(5),
        Paragraph("Deep Learning Project", TITLE),
        Paragraph("MLP, CNN and Sequence Models (RNN / LSTM / GRU / Seq2Seq)", SUBTITLE),
        sp(3),
        Paragraph("Bakouch Moad", AUTHOR),
        sp(2),
        Paragraph("EMSI Casablanca, Computer Science Department", COVER),
        Paragraph("Academic year 2025-2026", COVER),
        Paragraph("Module: Deep Learning", COVER),
        Paragraph("PyTorch implementation, CPU execution", COVER),
        PageBreak(),
    ]

    # Table of contents
    toc = TableOfContents()
    toc.levelStyles = [
        pstyle("TOC0", fontName="Helvetica-Bold", fontSize=11, leading=18,
               textColor=C_DARK, leftIndent=0),
        pstyle("TOC1", fontName="Helvetica", fontSize=10, leading=15,
               textColor=C_DARK, leftIndent=18),
    ]
    story += [Paragraph("Table of contents", H1PLAIN), sp(0.5), toc, PageBreak()]

    # Table of figures
    figlist = FigureList()
    figlist.levelStyles = [
        pstyle("FIG0", fontName="Helvetica", fontSize=9.5, leading=15,
               textColor=C_DARK, leftIndent=0),
    ]
    story += [Paragraph("Table of figures", H1PLAIN), sp(0.5), figlist, PageBreak()]

    # Introduction
    story += [
        Paragraph("General introduction", H1PLAIN),
        para("This report presents the design, implementation, experimentation and "
             "critical analysis of three families of deep learning models, each matched "
             "to a different data structure. The central goal is to show how a neural "
             "network architecture must follow the geometry and the statistical "
             "dependencies of the data it processes: independent tabular variables, "
             "images with strong spatial coherence, and sequences where order carries "
             "meaning."),
        para("The project is organised in three independent but complementary parts, "
             "implemented entirely in PyTorch:"),
        bullet("Part I: a multi-layer perceptron (MLP) for the classification of real "
               "tabular data (Breast Cancer Wisconsin)."),
        bullet("Part II: a convolutional neural network (CNN) for grayscale image "
               "classification (Fashion-MNIST)."),
        bullet("Part III: recurrent models (RNN, LSTM, GRU) and an encoder-decoder "
               "(Seq2Seq) architecture for sequence modelling and French to English "
               "translation."),
        para("Each part follows the same method: a reminder of the theoretical "
             "foundations, careful data preparation, a documented implementation, a "
             "comparative experiment, then a critical reading of the results and an "
             "answer to a synthesis question. A final cross-cutting question links the "
             "three approaches."),
        h2("Technical environment and methodology"),
        para("The whole code base follows standard PyTorch engineering practices:"),
        bullet("strict train / validation / test separation, with normalisation fitted "
               "only on the training data to avoid information leakage;"),
        bullet("correct use of model.train() and model.eval(), with torch.no_grad() "
               "during validation and test;"),
        bullet("hyper-parameters held in named variables, fixed random seed for "
               "reproducibility;"),
        bullet("systematic saving of figures and of the best models with torch.save."),
        sp(0.4),
        tcap("Table 1. Execution environment."),
        info_table([
            ("Language", "Python 3.13"),
            ("Framework", "PyTorch 2.11.0 with Torchvision"),
            ("Acceleration", "CPU"),
            ("Libraries", "NumPy, pandas, scikit-learn, seaborn, matplotlib"),
            ("Random seed", "42"),
        ]),
        PageBreak(),
    ]

    # ---------------- PART I ----------------
    story += [
        part_header("PART I. MLP and PyTorch engineering (tabular data)", H1BLUE),
        h2("1.1 Dataset, task and objectives"),
        para("The first part addresses the binary classification of real tabular data "
             "with a multi-layer perceptron. The dataset is Breast Cancer Wisconsin, a "
             "standard benchmark of 569 samples described by 30 numeric features computed "
             "from digitised images of cell nuclei (radius, texture, perimeter, area, "
             "smoothness and so on). The task is to predict whether a tumour is malignant "
             "or benign."),
        para("The dataset is clean (no missing values, no duplicate rows) but mildly "
             "imbalanced, with roughly 37 percent malignant and 63 percent benign cases. "
             "This imbalance is handled with a class-weighted loss so that the minority "
             "class is not neglected."),
        sp(0.3),
        tcap("Table 2. Summary of the Breast Cancer Wisconsin dataset."),
        info_table([
            ("Source", "sklearn.datasets.load_breast_cancer (no download)"),
            ("Samples", "569"),
            ("Features", "30 numeric"),
            ("Task", "binary classification (malignant / benign)"),
            ("Class balance", "212 malignant, 357 benign"),
            ("Quality", "0 missing values, 0 duplicate rows"),
            ("Split", "stratified 70 / 15 / 15 (398 / 85 / 86)"),
        ]),
        h2("1.2 Theoretical foundations (PyTorch core objects)"),
        para("The project opens with a reminder of the fundamental PyTorch objects "
             "needed throughout:"),
        bullet("nn.Module: base class of every network; it manages parameters, "
               "sub-modules, the train and eval modes and device placement."),
        bullet("Parameters: trainable tensors (requires_grad=True) tracked by the "
               "autograd engine."),
        bullet("Gradient: derivative of the loss with respect to each parameter, "
               "computed by backpropagation and stored in param.grad."),
        bullet("state_dict: dictionary mapping names to tensors; the object used for "
               "saving and reloading a model."),
        bullet("Device: compute hardware; here the CPU. Model and data must live on the "
               "same device."),
        para("The role of each layer is also recalled: nn.Linear (affine transform "
             "y = Wx + b), nn.ReLU (non-linearity max(0, x)) and nn.Dropout "
             "(regularisation that randomly zeroes activations during training)."),
        h2("1.3 Data preparation"),
        para("Preparation chains loading, validation, the stratified split, "
             "standardisation and the wrapping into PyTorch loaders. The scaler is "
             "fitted on the training set only."),
        code("raw = load_breast_cancer()\n"
             "df = pd.DataFrame(raw.data, columns=raw.feature_names)\n"
             "df['target'] = raw.target\n"
             "scaler = StandardScaler().fit(X_train)\n"
             "X_train = scaler.transform(X_train)",
             label="Block: loading and standardisation."),
        term("Duplicate rows: 0\n"
             "Class distribution: {1: 357, 0: 212}\n"
             "X_train: torch.Size([398, 30])  X_val: torch.Size([85, 30])  "
             "X_test: torch.Size([86, 30])\n"
             "Class weights (malignant, benign): tensor([1.3446, 0.7960])",
             label="Output: validation and split."),
        para("The class weights are computed as N / (2 * class_count) and passed to "
             "CrossEntropyLoss, which increases the cost of misclassifying the malignant "
             "(minority) class."),
        h2("1.4 Two MLP implementations"),
        para("The MLP is implemented in two equivalent ways, as required. The first "
             "stacks the layers declaratively in nn.Sequential. The second is "
             "object-oriented, inherits from nn.Module and builds its hidden layers "
             "dynamically from a list, which makes architecture sweeps easy. Both "
             "implement the same 30 -> 64 -> 32 -> 2 design with ReLU and Dropout(0.3)."),
        code("class MLP(nn.Module):\n"
             "    def __init__(self, in_dim=30, hidden=[64, 32], out_dim=2, p=0.3):\n"
             "        super().__init__()\n"
             "        layers, prev = [], in_dim\n"
             "        for h in hidden:\n"
             "            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(p)]\n"
             "            prev = h\n"
             "        layers.append(nn.Linear(prev, out_dim))\n"
             "        self.net = nn.Sequential(*layers)\n"
             "    def forward(self, x):\n"
             "        return self.net(x)",
             label="Block: custom MLP (subclass of nn.Module)."),
        h2("1.5 Parameter inspection"),
        para("Inspecting the model with named_parameters and state_dict confirms the "
             "dimension of each layer. The 30 -> 64 -> 32 -> 2 architecture totals 4130 "
             "trainable parameters, identical for both implementations."),
        term("Sequential(\n"
             "  (0): Linear(in_features=30, out_features=64)\n"
             "  (1): ReLU()\n"
             "  (2): Dropout(p=0.3)\n"
             "  (3): Linear(in_features=64, out_features=32)\n"
             "  (4): ReLU()\n"
             "  (5): Dropout(p=0.3)\n"
             "  (6): Linear(in_features=32, out_features=2))\n"
             "Total parameters: 4130",
             label="Output: parameter inspection."),
        h2("1.6 Initialisation strategies"),
        para("Four initialisation strategies are compared on the same architecture: "
             "Gaussian N(0, 0.01), constant (0.1), Xavier uniform and Kaiming normal. "
             "Xavier scales the weights by sqrt(6 / (fan_in + fan_out)) and keeps the "
             "activation variance stable; Kaiming is tuned for ReLU networks. On this "
             "small and well-conditioned problem the four strategies converge to similar "
             "test accuracy (0.954 to 0.965), with Kaiming and Gaussian reaching the best "
             "result and constant initialisation the weakest."),
        sp(0.3),
        tcap("Table 3. Test metrics by initialisation strategy."),
        result_table(
            ["Strategy", "Accuracy", "Precision", "Recall", "F1-score"],
            [["Gaussian (0.01)", "0.9651", "0.9811", "0.9630", "0.9720"],
             ["Constant (0.1)", "0.9535", "0.9630", "0.9630", "0.9630"],
             ["Xavier uniform", "0.9535", "0.9808", "0.9444", "0.9623"],
             ["Kaiming normal", "0.9651", "0.9811", "0.9630", "0.9720"]],
        ),
        figure("loss_curves_kaiming.png",
               "Training and validation loss for the Kaiming initialisation, "
               "among the best of the four strategies."),
        h2("1.7 Final training, saving and device check"),
        para("The reference model (Sequential MLP, Xavier initialisation) is trained "
             "with Adam (lr=1e-3, weight_decay=1e-4) and early stopping on the "
             "validation loss (patience 10). The validation accuracy reaches about 0.977 "
             "during training. The best checkpoint, selected on the validation loss, is "
             "saved with torch.save."),
        term("Epoch   1 | Train Loss: 0.5694 Acc: 0.7362 | Val Loss: 0.3197 Acc: 0.8941\n"
             "Epoch  10 | Train Loss: 0.0930 Acc: 0.9673 | Val Loss: 0.0780 Acc: 0.9765\n"
             "Epoch  30 | Train Loss: 0.0449 Acc: 0.9849 | Val Loss: 0.0521 Acc: 0.9765\n"
             "Epoch  50 | Train Loss: 0.0327 Acc: 0.9899 | Val Loss: 0.0437 Acc: 0.9765\n"
             "Early stopping at epoch 51 (patience=10)",
             label="Output: training log."),
        figure("loss_curves_sequential.png",
               "Training and validation loss of the final Sequential MLP."),
        para("The best model is saved and reloaded with load_state_dict. The predictions "
             "before and after reloading are identical, which confirms that state_dict "
             "captures the full state of the model."),
        term("--- Test (Reloaded model) Metrics ---\n"
             "Accuracy : 0.9651   Precision: 0.9811   Recall: 0.9630   F1-Score: 0.9720\n"
             "Reload successful, identical predictions confirmed.",
             label="Output: save and reload check."),
        h2("1.8 Final evaluation on the test set"),
        para("On the held-out test set (86 samples never seen during training or model "
             "selection) the Sequential MLP reaches:"),
        sp(0.2),
        tcap("Table 4. Test metrics of the MLP."),
        result_table(
            ["Metric", "Value"],
            [["Accuracy", "0.9651"], ["Precision", "0.9811"],
             ["Recall", "0.9630"], ["F1-score", "0.9720"]],
            widths=[CONTENT_W * 0.5, CONTENT_W * 0.5],
        ),
        figure("confusion_matrix_sequential.png",
               "Confusion matrix of the MLP on the test set.",
               width=11 * cm),
        para("Recall matters most here: a missed malignancy is the costliest error in a "
             "clinical setting. The confusion matrix shows the model misclassifies only 1 "
             "of the 32 malignant cases, which is why the loss is weighted in favour of "
             "the malignant class."),
        h2("1.9 Synthesis question, Part I"),
        question("To what extent is a well-tuned MLP a relevant solution for tabular "
                 "classification on a real dataset, and what are its main limits with "
                 "respect to the statistical structure of the data?"),
        para("The MLP is relevant: as a universal approximator (Cybenko theorem) it "
             "reaches about 96.5 percent accuracy on Breast Cancer, with high recall on "
             "the malignant class. Its success rests on careful preprocessing, in "
             "particular standardisation, without which features measured on very "
             "different scales would dominate the gradient. Regularisation (Dropout, "
             "weight decay), the Adam optimiser and the class-weighted loss give a stable "
             "convergence, and the choice of initialisation has only a minor effect on "
             "this small, well-conditioned problem."),
        para("Its limits come from the statistical structure of the data: the MLP is "
             "invariant to permutations of its inputs, so it cannot natively exploit "
             "structured interactions between variables, a domain where tree-based models "
             "(gradient boosting) are often superior on tabular data. It also depends "
             "heavily on feature scaling and offers limited interpretability."),
        PageBreak(),
    ]

    # ---------------- PART II ----------------
    cnn = read_csv("cnn_comparison.csv")
    name_map = {
        "baseline_lenet": "Baseline LeNet (avg pool, pad=2)",
        "padding_valid": "Valid padding (pad=0)",
        "stride2": "Stride 2 in conv1",
        "maxpool": "Max pooling",
        "filters_half": "Half filters",
        "filters_double": "Double filters",
        "with_1x1": "With 1x1 convolution",
        "mlp_baseline": "MLP baseline (flatten)",
    }
    cnn_rows = [[name_map.get(r["Experiment"], r["Experiment"]),
                 "{:,}".format(int(r["Params"])),
                 r["Best Test Acc"], r["Train Time (s)"]] for r in cnn]

    story += [
        part_header("PART II. CNN and computer vision (images)", H1TEAL),
        h2("2.1 Dataset, task and objectives"),
        para("The second part addresses grayscale image classification with a "
             "convolutional neural network. The dataset is Fashion-MNIST, made of 60000 "
             "training and 10000 test images of 28 by 28 pixels in 10 clothing classes "
             "(T-shirt, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, "
             "ankle boot). It is a drop-in but harder replacement for the digits of "
             "MNIST, on which a small LeNet-style network plateaus around 85 to 92 "
             "percent."),
        sp(0.3),
        tcap("Table 5. Summary of the Fashion-MNIST dataset."),
        info_table([
            ("Source", "torchvision.datasets.FashionMNIST (auto-download)"),
            ("Size", "60000 train, 10000 test"),
            ("Image", "28 x 28, 1 channel (grayscale)"),
            ("Classes", "10 clothing categories"),
            ("Normalisation", "mean 0.2860, std 0.3530"),
            ("Augmentation", "random horizontal flip, random crop (pad=4) on train"),
        ]),
        figure("fashion_mnist_samples.png",
               "Sample images from Fashion-MNIST, one per class.",
               width=13 * cm),
        h2("2.2 Why the MLP is inadequate for images"),
        para("Three limits justify moving from the MLP to the CNN on images:"),
        bullet("Parameter explosion: a fully connected layer over a flattened image "
               "needs one weight per pixel and per neuron, which grows very fast."),
        bullet("Loss of spatial structure: flattening scatters neighbouring pixels, so "
               "the MLP has to relearn the 2D topology from scratch."),
        bullet("No invariance: an object shifted by a few pixels becomes a completely "
               "different input for the MLP."),
        para("The CNN answers with locality (k by k kernels), weight sharing "
             "(equivariance to translation, few parameters), a hierarchy of "
             "representations (edges, then textures, then objects) and pooling "
             "(invariance to small shifts)."),
        h2("2.3 Dimensional computations"),
        para("The output size formulas are recalled and applied. For a convolution, "
             "H_out = (H_in + 2P - K) / S + 1; for a pooling, H_out = (H_in - K) / S + 1. "
             "The full dimensional trace of the network is:"),
        sp(0.3),
        tcap("Table 6. Dimensional trace of the LeNet-style CNN."),
        result_table(
            ["Layer", "Operation", "Params", "Output (C x H x W)"],
            [["Input", "-", "0", "1 x 28 x 28"],
             ["Conv1", "Conv2d(1->6, k=5, pad=2)", "156", "6 x 28 x 28"],
             ["Pool1", "AvgPool2d(2, 2)", "0", "6 x 14 x 14"],
             ["Conv2", "Conv2d(6->16, k=5)", "2,416", "16 x 10 x 10"],
             ["Pool2", "AvgPool2d(2, 2)", "0", "16 x 5 x 5"],
             ["FC1", "Linear(400->120)", "48,120", "120"],
             ["FC2", "Linear(120->84)", "10,164", "84"],
             ["FC3", "Linear(84->10)", "850", "10"]],
            widths=[1.7 * cm, 5.6 * cm, 2.3 * cm, CONTENT_W - 9.6 * cm],
        ),
        para("The network totals 61,706 parameters."),
        h2("2.4 Manual implementations versus PyTorch"),
        para("To anchor the theory, 2D cross-correlation and max pooling are "
             "reimplemented by hand in NumPy and compared with the PyTorch operators. "
             "The key cell is:"),
        code("def cross_correlate2d(X, K):\n"
             "    kH, kW = K.shape\n"
             "    H, W = X.shape\n"
             "    out = np.zeros((H - kH + 1, W - kW + 1))\n"
             "    for i in range(out.shape[0]):\n"
             "        for j in range(out.shape[1]):\n"
             "            out[i, j] = (X[i:i+kH, j:j+kW] * K).sum()\n"
             "    return out",
             label="Block: manual 2D cross-correlation."),
        term("=== Manual Convolution Verification ===\n"
             "Input shape:   (6, 6)   Kernel shape: (3, 3)   Output shape: (4, 4)\n"
             "Max diff vs PyTorch: 2.31e-07  (PASS)\n"
             "Conv1 (1->6, k=5): 156 params   Conv2 (6->16, k=5): 2416 params\n"
             "1x1 conv (6->16): 112 params",
             label="Output: manual versus PyTorch check."),
        para("The manual results match PyTorch to floating-point precision (maximum "
             "difference 2.31e-07), which validates the understanding of the operators."),
        h2("2.5 Architecture (LeNet) and training"),
        para("The CNN follows LeNet-5: two convolution blocks (Conv, ReLU, pooling) "
             "followed by three fully connected layers. The baseline uses average "
             "pooling and pad=2 on the first convolution."),
        code("self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)\n"
             "self.pool1 = nn.AvgPool2d(2, 2)\n"
             "self.conv2 = nn.Conv2d(6, 16, kernel_size=5)\n"
             "self.pool2 = nn.AvgPool2d(2, 2)\n"
             "self.fc1 = nn.Linear(400, 120)\n"
             "self.fc2 = nn.Linear(120, 84)\n"
             "self.fc3 = nn.Linear(84, 10)",
             label="Block: LeNet-style CNN."),
        para("Trained for 20 epochs with Adam (lr=1e-3), the baseline reaches 88.8 "
             "percent test accuracy."),
        figure("lenet_baseline_loss.png",
               "Training loss of the baseline LeNet on Fashion-MNIST."),
        h2("2.6 Ablation study"),
        para("Eight configurations are compared, each changing one factor while keeping "
             "the rest fixed, to isolate the effect of padding, stride, pooling type, "
             "filter count and the 1x1 convolution. A flatten-based MLP is added as a "
             "reference."),
        sp(0.3),
        tcap("Table 7. Ablation study on Fashion-MNIST (20 epochs)."),
        result_table(
            ["Configuration", "Params", "Best test acc", "Time (s)"],
            cnn_rows,
            widths=[CONTENT_W - 8.2 * cm, 3.0 * cm, 2.6 * cm, 2.6 * cm],
        ),
        figure("cnn_experiment_comparison.png",
               "Test accuracy across the CNN configurations."),
        para("Reading the results: switching to max pooling raises accuracy from 88.8 to "
             "89.2 percent, because it keeps the strongest activation (the response of "
             "an edge detector) rather than averaging it away. Doubling the filters gives "
             "the best result (89.8 percent) at the cost of roughly twice the parameters "
             "and training time, while halving them lowers accuracy to 85.6 percent. A "
             "stride of 2 sub-samples too aggressively on a 28 by 28 input and is the "
             "weakest CNN variant (82.2 percent). Valid padding stays close to the "
             "baseline, and the 1x1 convolution adds cross-channel mixing at almost no "
             "spatial cost."),
        h2("2.7 Feature map visualisation (hooks)"),
        para("The activations of the first convolution are captured with "
             "register_forward_hook, without modifying the forward pass."),
        code("def hook(name):\n"
             "    def fn(m, i, o):\n"
             "        activation[name] = o.detach()\n"
             "    return fn\n"
             "model.conv1.register_forward_hook(hook('conv1'))",
             label="Block: capturing activations with a hook."),
        figure("feature_maps_conv1.png",
               "Feature maps of the first convolution for one test image. "
               "Each filter responds to a different local pattern.",
               width=15.5 * cm, max_h=4 * cm),
        h2("2.8 MLP versus CNN comparison"),
        para("A flatten-based MLP is trained on the same images to measure the "
             "contribution of the convolutional structure."),
        sp(0.2),
        tcap("Table 8. MLP versus CNN on Fashion-MNIST."),
        result_table(
            ["Model", "Params", "Best test acc"],
            [["MLP (flatten, 784->256->128->10)", "235,146", "0.8439"],
             ["CNN baseline (LeNet)", "61,706", "0.8875"],
             ["CNN best (double filters)", "117,078", "0.8975"]],
            widths=[CONTENT_W - 6.6 * cm, 3.3 * cm, 3.3 * cm],
        ),
        para("The baseline CNN beats the MLP while using about 3.8 times fewer "
             "parameters, and the best CNN variant extends the gap further. This "
             "advantage comes directly from the inductive biases suited to images: "
             "locality and weight sharing."),
        h2("2.9 Synthesis question, Part II"),
        question("Why is a CNN more relevant than an MLP for image classification, and "
                 "how do padding, stride, pooling and depth actually influence "
                 "performance?"),
        para("The CNN is structurally better suited to images: its inductive biases "
             "(locality, weight sharing, hierarchy) exploit the 2D geometry that the MLP "
             "destroys by flattening. With fewer parameters the CNN reaches higher "
             "accuracy (88.8 percent for the baseline against 84.4 percent for a far "
             "larger MLP, and up to 89.8 percent with more filters)."),
        para("The architectural choices have a concrete and measurable effect, confirmed "
             "by the ablation: padding preserves border information and spatial size; a "
             "larger stride sub-samples and loses detail; max pooling captures the "
             "presence of a pattern better than average pooling; more filters increase "
             "capacity (useful here) but demand more training and compute; the 1x1 "
             "convolution recombines channels cheaply."),
        PageBreak(),
    ]

    # ---------------- PART III ----------------
    rnn = read_csv("rnn_comparison.csv")
    rnn_rows = [[r["Cell"], "{:,}".format(int(r["Params"])),
                 r["Final Train PPL"], r["Final Val PPL"], r["Train Time (s)"]]
                for r in rnn]

    story += [
        part_header("PART III. RNN, LSTM, GRU and Seq2Seq (sequences)", H1PURPLE),
        h2("3.1 Theoretical foundations"),
        para("This part covers two complementary sequence tasks: a comparison of three "
             "recurrent cells (RNN, LSTM, GRU) on next-token prediction, measured by "
             "perplexity, and an encoder-decoder (Seq2Seq) system for French to English "
             "translation, evaluated by exact accuracy and BLEU."),
        para("The sequence model is probabilistic. By the chain rule, the probability of "
             "a sequence is the product of conditional probabilities of each token given "
             "the previous ones, and training minimises the negative log-likelihood "
             "(cross-entropy). Perplexity, defined as exp(loss), measures the average "
             "number of equally likely choices the model hesitates between at each step "
             "(1 is perfect). The notebook also recalls the simple RNN "
             "(h_t = tanh(W_h h_(t-1) + W_x x_t + b)) and its gradient problem, the LSTM "
             "(three gates plus an additive cell state that carries information), the GRU "
             "(two gates, lighter), backpropagation through time, and gradient clipping "
             "(rescale the gradient when its norm exceeds a threshold)."),
        h2("3.2 Data preparation (sequences)"),
        para("The task uses a compact French to English parallel corpus of 2000 sentence "
             "pairs built from common everyday phrases. Each sentence is tokenised at the "
             "word level, with the special tokens pad, sos, eos and unk. The corpus is "
             "split 80 / 10 / 10 and the vocabularies are built on the training set only. "
             "Batches are padded to the longest sequence in the batch with a custom "
             "collate function."),
        term("Synthetic corpus: 2000 pairs\n"
             "Src vocab size: 25, Tgt vocab size: 27\n"
             "Train: 1600, Val: 200, Test: 200\n"
             "Batch src shape: torch.Size([32, 5]), tgt shape: torch.Size([32, 6])",
             label="Output: corpus preparation."),
        h2("3.3 Three recurrent models (RNN, LSTM, GRU)"),
        para("The three models share the same architecture, an embedding followed by a "
             "recurrent layer and a linear projection; only the cell type changes, which "
             "makes the comparison fair."),
        code("class RecurrentModel(nn.Module):\n"
             "    def __init__(self, vocab_size, emb_dim=128, hid_dim=256,\n"
             "                 n_layers=2, dropout=0.5, cell='lstm'):\n"
             "        super().__init__()\n"
             "        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)\n"
             "        Cell = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}[cell]\n"
             "        self.rnn = Cell(emb_dim, hid_dim, num_layers=n_layers,\n"
             "                        dropout=dropout, batch_first=True)\n"
             "        self.fc = nn.Linear(hid_dim, vocab_size)",
             label="Block: generic recurrent model."),
        h2("3.4 BPTT and gradient clipping"),
        para("During training the gradient norm is tracked and clipped with "
             "clip_grad_norm_ at a maximum norm of 1.0. The next figure shows the "
             "gradient norms of the LSTM across training steps: the spikes above the "
             "threshold are clipped, which keeps the updates bounded and the training "
             "stable, the precise role of gradient clipping in recurrent networks."),
        figure("gradient_norms_lstm.png",
               "LSTM gradient norms across training steps, with the clipping threshold."),
        h2("3.5 RNN / LSTM / GRU comparison"),
        para("The three models are trained for 5 epochs on the corpus. The table reports "
             "the final perplexity, the parameter count and the training time."),
        sp(0.3),
        tcap("Table 9. Comparison of the recurrent cells (next-token prediction)."),
        result_table(
            ["Cell", "Params", "Train PPL", "Val PPL", "Time (s)"],
            rnn_rows,
            widths=[2.6 * cm, 3.4 * cm, CONTENT_W - 12.0 * cm, 3.0 * cm, 3.0 * cm],
        ),
        figure("rnn_lstm_gru_perplexity.png",
               "Training and validation perplexity for RNN, LSTM and GRU."),
        para("On this short corpus the three cells converge to a similar low perplexity "
             "(around 2.0), because the dependencies are short enough that the simple RNN "
             "is not penalised. The simple RNN is also the cheapest in parameters and "
             "time. The advantage of the gated cells (LSTM, GRU) would grow on longer "
             "sequences, where the additive cell state of the LSTM and the gates of the "
             "GRU preserve information over more steps."),
        h2("3.6 Seq2Seq with teacher forcing"),
        para("The Seq2Seq model splits source reading and target generation into an "
             "encoder and a decoder, both LSTM. The encoder compresses the French "
             "sentence into a context (its final hidden and cell states); the decoder "
             "then generates the English tokens one by one. Teacher forcing (ratio 0.5) "
             "feeds the ground-truth previous token with probability 0.5 and the model's "
             "own prediction otherwise, which stabilises early training."),
        code("def forward(self, src, tgt, teacher_forcing_ratio=0.5):\n"
             "    B, T_tgt = tgt.shape\n"
             "    h, c = self.encoder(src)\n"
             "    outputs = torch.zeros(B, T_tgt - 1, vocab_size)\n"
             "    input_t = tgt[:, 0]\n"
             "    for t in range(T_tgt - 1):\n"
             "        logit, h, c = self.decoder(input_t, h, c)\n"
             "        outputs[:, t, :] = logit\n"
             "        use_teacher = torch.rand(1).item() < teacher_forcing_ratio\n"
             "        input_t = tgt[:, t + 1] if use_teacher else logit.argmax(-1)\n"
             "    return outputs",
             label="Block: Seq2Seq forward pass with teacher forcing."),
        para("The full model has about 1.86 million parameters. Training drives the "
             "validation perplexity down to about 1.01, which means the model has learned "
             "the alignment between the two languages on this corpus."),
        figure("seq2seq_perplexity.png",
               "Training and validation perplexity of the Seq2Seq model."),
        h2("3.7 Decoding (greedy versus beam) and BLEU"),
        para("Two decoding strategies are compared: greedy decoding (pick the most "
             "likely token at each step) and beam search (keep the three best partial "
             "hypotheses). On the test set both reach a BLEU of 100 and exact "
             "translations."),
        term("SRC: ou est la gare        REF: where is the station   HYP: where is the station\n"
             "SRC: merci beaucoup        REF: thank you very much    HYP: thank you very much\n"
             "SRC: je parle francais     REF: i speak french         HYP: i speak french\n"
             "BLEU (greedy, k=1): 100.00\n"
             "BLEU (beam,   k=3): 100.00",
             label="Output: sample translations and BLEU."),
        sp(0.2),
        tcap("Table 10. Seq2Seq evaluation on the test set."),
        result_table(
            ["Strategy", "Exact accuracy", "BLEU"],
            [["Greedy (k=1)", "100 percent", "100.0"],
             ["Beam search (k=3)", "100 percent", "100.0"]],
            widths=[CONTENT_W - 7.0 * cm, 3.5 * cm, 3.5 * cm],
        ),
        para("Both strategies reach 100 percent. Since the task is fully learned on this "
             "corpus, greedy decoding is already optimal and beam search matches it "
             "without surpassing it. The benefit of beam search appears when the model is "
             "uncertain and an early greedy mistake cannot be recovered, which does not "
             "happen here."),
        h2("3.8 Synthesis question, Part III"),
        question("How well do recurrent architectures model a sequence, and what "
                 "justifies moving from a simple RNN to an LSTM or GRU and then to an "
                 "encoder-decoder scheme?"),
        para("Recurrent architectures model sequences by propagating a state through "
             "time. Moving from the simple RNN to the LSTM or GRU is justified by the "
             "gradient problem of backpropagation through time: the gated cells preserve "
             "information over longer distances, and gradient clipping keeps training "
             "stable. For a transduction between sequences of different lengths (a French "
             "sentence to its English translation) a single recurrent model is not "
             "enough: the encoder-decoder scheme separates reading from generation, "
             "teacher forcing stabilises learning, and beam search matches or improves "
             "greedy decoding. The remaining limits, the bottleneck of a single context "
             "vector and the poorly parallelisable sequential computation, motivate "
             "attention and then the Transformer."),
        PageBreak(),
    ]

    # ---------------- Transversal + Conclusion ----------------
    story += [
        part_header("Final cross-cutting question", H1DARK),
        question("How does deep learning adapt its architectures to the structure of the "
                 "data (tabular, image and sequential), and why must a single supervised "
                 "learning paradigm be declined differently according to the geometry, "
                 "the local dependency, the temporality and the representation of the "
                 "data?"),
        para("The three parts share a single paradigm: minimise a loss by gradient "
             "descent (forward, loss, backward, step). What changes is the inductive bias "
             "encoded in the architecture, chosen according to the structure of the data:"),
        bullet("Tabular (MLP): no privileged geometry or order between variables, so "
               "dense layers; the effort goes into preprocessing (standardisation)."),
        bullet("Image (CNN): strong local dependency and translation invariance, so "
               "convolution (locality and weight sharing) and pooling, which exploit the "
               "2D geometry with few parameters."),
        bullet("Sequence (RNN, Seq2Seq): temporality and variable length, so recurrent "
               "architectures that carry a state, completed by the encoder-decoder scheme "
               "for transduction."),
        para("An MLP applied to an image would explode in parameters and lose the spatial "
             "structure (Part II shows 84.4 percent against 88.8 to 89.8 percent for the "
             "CNN); a CNN on text would ignore long-range dependencies; an RNN on tabular "
             "data would invent a non-existent order. The shape of the model must follow "
             "the shape of the data: this is the guiding principle that links the three "
             "parts of the project."),
        sp(0.6),
        part_header("General conclusion", H1DARK),
        para("This project implemented, end to end and in PyTorch, three families of deep "
             "learning models on three distinct data structures. The experimental "
             "results, all reproducible, confirm the central thesis: the match between "
             "architecture and data structure is decisive."),
        sp(0.3),
        tcap("Table 11. Summary of the project results."),
        result_table(
            ["Part", "Dataset", "Model", "Main result"],
            [["I, MLP", "Breast Cancer", "MLP 30->64->32->2", "96.5 percent test accuracy"],
             ["II, CNN", "Fashion-MNIST", "LeNet (avg/max pool)", "88.8 to 89.8 percent vs MLP 84.4"],
             ["III, LM", "FR-EN corpus", "RNN / LSTM / GRU", "val perplexity about 2.0"],
             ["III, Seq2Seq", "FR-EN translation", "encoder-decoder LSTM", "100 percent, BLEU 100"]],
            widths=[2.6 * cm, 3.6 * cm, 4.2 * cm, CONTENT_W - 10.4 * cm],
        ),
        para("Beyond the raw performance, the project covered PyTorch engineering "
             "(nn.Module, parameter and device handling, saving and reloading, hooks), "
             "initialisation and regularisation strategies, ablation experiments, and the "
             "mechanisms specific to sequences (backpropagation through time, gradient "
             "clipping, teacher forcing, decoding). The natural extensions are attention "
             "and the Transformer architectures, which lift the limits of recurrent "
             "models."),
    ]

    flat = []
    for item in story:
        if isinstance(item, (list, tuple)):
            flat.extend(item)
        else:
            flat.append(item)

    frame = Frame(2 * cm, 2 * cm, CONTENT_W, H - 4.2 * cm, id="body")
    doc = ReportDoc(OUT_PDF, pagesize=A4, title="Deep Learning Project",
                    author="Bakouch Moad")
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=on_page)])
    doc.multiBuild(flat)
    print("PDF written to %s" % OUT_PDF)


if __name__ == "__main__":
    build()
