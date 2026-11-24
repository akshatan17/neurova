# Viva preparation: likely questions and answers

Read these until you can answer them in your own words without looking. The numbers come from the trained models in this project.

## The problem

**Why does this need deep learning? Couldn't you just threshold the bright region?**
MRI intensity has no fixed scale. It changes with the scanner, the field strength, the sequence and the settings, so any threshold that works on one machine fails on another. The three tumour types also differ in ways that are about shape, texture and position, such as a meningioma pressing in from the skull surface or a pituitary tumour sitting at the sella. A CNN learns those features from labelled examples. For sleep, the textbook rules exist, but real EEG is noisy and the right stage for one epoch depends on its neighbours, which is exactly what a recurrent network models.

**Who would actually use this?**
Diagnostic centres in smaller cities that have an MRI scanner but no neuroradiologist, and the teleradiology firms that read for them. The model reorders the reading list so scans that probably show a tumour are read first. It never reports on its own.

## Data

**Where did the data come from, and is it legal to use?**
The Brain Tumor MRI Dataset (Nickparvar, 2021, CC BY 4.0), 7,023 slices combined from figshare, SARTAJ and Br35H, and Sleep-EDF Expanded from PhysioNet (ODC Attribution licence). Both are public, de-identified and licensed for reuse with citation.

**What did you find when you checked the data?**
{removed} of the {total} MRI images were duplicates, found with a perceptual hash, and {cross} duplicate groups appeared in both the official Training and Testing folders. If you evaluate on the official split, part of your test set is images the model trained on, so the score is inflated. I removed the duplicates and made a fresh stratified 70/15/15 split.

**What is a perceptual hash?**
A short fingerprint of what an image looks like rather than its exact bytes. Images that are the same picture saved at a different size or compression get the same or a very similar hash, so they're caught where a file checksum would miss them.

**Why split the sleep data by person?**
Epochs from the same night look alike. If you split epochs at random, the same person's brain activity is in both training and test, and the model can score well by recognising the person. Splitting by subject tests the real question: does it work on someone new?

**How is the test set kept honest?**
Validation chose the best epoch, fitted the calibration temperature and chose which model to deploy. The test set was used once, at the very end.

## The MRI model

**Why EfficientNet-B0?**
It gives about the best ImageNet accuracy per unit of computation among standard networks, at 4 million parameters, which matters because the app runs on a CPU. Pretraining helps because early filters that detect edges and textures transfer well, and we have thousands of images, not millions.

**What does fine-tuning mean here?**
I loaded the official ImageNet weights, replaced the 1,000-class output layer with a new 4-class layer, and trained the whole network with a small learning rate (0.0003) so the pretrained features are adjusted rather than destroyed.

**How did the baseline compare?**
NeurovaCNN, trained from scratch, scored {cnn_acc} test accuracy against {eff_acc} for EfficientNet. That gap is the benefit of transfer learning.

**Why macro-F1 and not accuracy?**
Macro-F1 averages the F1 score of each class equally, so a model can't look good by doing well on the biggest class and badly on a small one.

**What is label smoothing?**
Instead of training towards 100% for the right class and 0% for the others, the target is slightly softened (90% plus a share of the remaining 10%). It discourages extreme, over-confident outputs.

**What is temperature scaling, and what was your temperature?**
After training, the logits are divided by one number T before the softmax, with T fitted on validation data. It changes how confident the model is without changing which class wins. My T was {temp}. Because it is below 1, it sharpened the probabilities: label smoothing had made the model slightly under-confident. Calibration error fell from {ece_b} to {ece_a}.

**What is Grad-CAM?**
It takes the gradient of the predicted class score with respect to the last convolutional feature maps, averages each map's gradient to get a weight, and sums the weighted maps. Bright regions are the parts of the image that pushed the model towards its answer.

**Does a correct prediction mean the model looked at the tumour?**
No. In one correct pituitary case the heatmap sits low over the brainstem instead of the sella, and in one error, a pituitary tumour called meningioma, the heatmap is at the top of the brain, nowhere near the tumour. That's why the heatmap is shown: it lets a reviewer catch a prediction made for the wrong reason.

**What mistakes does it make?**
{mistakes} errors out of {n_test} test images. Almost all are confusions between tumour types, mostly gliomas read as meningiomas. {missed_text}

**Your accuracy is very high. Should we believe it?**
On this test set, yes, because duplicates were removed and the test set was used once. But the robustness test showed the standard model was fragile: noise of about three grey levels, invisible to a person, changed {flip_std} of its test predictions. That's a sign of shortcut learning. The four classes come from different source collections with different resolutions and compression, so the network could partly identify the source instead of the tumour. High accuracy on one dataset doesn't prove the model learned anatomy.

**What did you do about it?**
I retrained with augmentation that destroys those fingerprints: random down-and-up resampling, JPEG re-compression, blur and noise, applied at random. Flips under faint noise fell to {flip_rob}, and test accuracy was {rob_acc}. Before training it, I fixed the rule for which model to deploy: the scrubbed one if its validation macro-F1 is within one point of the best. {deploy_note} The real test would be scans from a different hospital, which is the first thing the pilot would measure.

**What happens if you give it a tumour it wasn't trained on?**
It gets it wrong, and I tested that. I gave it an MRI of an eye-socket tumour found online, and it said meningioma at 59%, with no tumour second. The model has only four possible answers, so it has to pick one of them. It learned where and how the four classes usually appear, not what a glioma is in general. Fixing that properly needs labelled scans of the new condition, which I don't have.

**So what did you do about it?**
I added a check for unfamiliar scans. Every scan becomes 1,280 features inside the network. I reduce those to {fam_k} with PCA and measure the Mahalanobis distance to the nearest class of training scans, which is distance counted in units of how much the training scans themselves vary. If it's beyond what 95% of validation scans reach, the app warns that the scan is unlike its training data. On the test set that flags {fam_caught} of the model's {mistakes} mistakes and {fam_false}% of correct answers, and it flagged the eye-socket scan. It's weak in other ways: it only catches {fam_updown}% of scans turned upside down. So it's a warning, not a guarantee.

**Why is the app's accuracy still 98.8% if it fails on that scan?**
Accuracy is measured on test scans of the four classes it was trained for. A scan from outside those classes isn't in that measurement at all. That's the difference between accuracy and scope, and it's why the report states what the model does not cover.

**What is Monte Carlo dropout?**
Dropout is left switched on at prediction time and the model is run 20 times. If the answers vary a lot, the model is uncertain about this scan. It's a cheap approximation to Bayesian uncertainty.

**What is the robustness check?**
The app re-runs the model on the scan seven more times, with added noise, blur and brightness changes, and reports whether the answer changes. An adversarial test such as FGSM, which nudges every pixel in the direction that most increases the loss, needs gradients, and the exported model in the app doesn't compute them, so that test is not in the app.

## The sleep model

**Explain the architecture.**
A 1-D CNN reads each 30-second epoch (3,000 samples at 100 Hz) and reduces it to 128 features. The first layer has a wide 0.5-second kernel, which works like a learned filter bank for EEG rhythms. Then a bidirectional GRU reads 20 consecutive epochs forwards and backwards, so each epoch's prediction can use the ten minutes around it, the way a human scorer checks context.

**Why a GRU and not an LSTM or a Transformer?**
A GRU has two gates where an LSTM has three plus a separate cell state, so it has fewer parameters and trains faster, which mattered on a CPU, and published sleep models (TinySleepNet, DeepSleepNet) show recurrent models work well here. For the Transformer I didn't just assume: I built one. Same CNN encoder, but a two-layer Transformer encoder with four attention heads and learned position embeddings reads the 20 epochs instead of the GRU. Its first run scored {trf_first}, above the GRU's {gru_first}, and I nearly reported that. But with two test people one run can be lucky, so I trained each version three times with different random seeds. The GRU averaged {gru_acc}, range {gru_rng}; the Transformer averaged {trf_acc}, range {trf_rng}. {trf_verdict} Validation macro-F1 agreed: {gru_val} for the GRU, {trf_val} for the Transformer. With 20 nights a Transformer has too many parameters and no built-in sense of order, so the GRU stayed.

**What's the difference between how a GRU and a Transformer read the sequence?**
The GRU reads the 20 epochs one at a time, forwards and backwards, carrying a memory, so information from far away has to survive many steps. The Transformer's self-attention lets every epoch look at every other epoch directly in one step and decide how much each matters. Attention has no sense of order by itself, which is why it needs position embeddings. It also has more parameters to learn, which is why it usually needs more data.

**Why did the same model score differently with a different seed?**
The seed sets the random starting weights and the order of the training batches. With only two people in the test set, small differences in what the model learned change the score by a few points. That's why I report the average and range over three seeds rather than one number.

**How good is kappa {kappa}?**
Cohen's kappa measures agreement beyond chance. Human scorers typically agree with each other at about 0.75 to 0.8, so {kappa} on unseen people is in that range.

**Which stage is hardest and why?**
N1 had the lowest F1 ({n1_f1}). It is brief and transitional, looks like both wake and N2, and is rare. Wake recall was also low ({w_rec}): the model often calls quiet wake N1. Class weights in the loss helped N1 but didn't solve it.

## The system

**Walk me through one request.**
The browser sends the file with a signed token. The server checks the token, the role and the patient's consent, checks the file, runs the same preprocessing function used in training (crop to the brain, resize to 192, normalise), runs the ONNX model, applies temperature scaling, computes Grad-CAM, uncertainty and the stability check, stores the result with the patient's record, writes an audit entry, and returns JSON that the page displays. The uploaded file itself is not kept.

**Why ONNX? Isn't this a PyTorch project?**
The models are trained in PyTorch. PyTorch is needed to train a network, because training needs gradients for millions of weights, but running a trained network only needs the forward pass. ONNX is a standard file format for a trained network, and ONNX Runtime is a small engine that runs it. Exporting cut the app's install from over 1 GB to about 100 MB and halved the time per scan. I checked the export two ways: the outputs match PyTorch to five decimal places, and the ONNX model scores the same {eff_acc_deployed} on all {n_test} test images.

**How do you get Grad-CAM without PyTorch's autograd?**
The exported model returns the last feature maps as well as the class scores. After those maps the network only applies the swish activation, averages each map, and applies one linear layer. That is simple enough to differentiate by hand: the gradient of a class score with respect to a feature map value is the linear layer's weight for that map, times the slope of swish at that value, divided by 36. I compute that in NumPy, and it matches PyTorch's Grad-CAM to six decimal places. Monte Carlo dropout works the same way, because the only dropout layer sits just before the linear layer.

**What does the clinician's review add?**
The model never has the last word. Every result is marked as waiting for review until a clinician the patient has shared with records whether they agree, with a note the patient can read. The admin page shows how often clinicians agreed, which is the start of monitoring the model in use.

**How does the triage list decide the order?**
Three groups. First, any MRI the model calls a tumour, most confident first, because those are the scans a radiologist should read today. Second, any result where a check failed: high or moderate uncertainty, a close call between two classes, an answer that changed under the stability test, or an unfamiliar scan. Those are the ones where the model may simply be wrong, like the glioma it called "no tumour" at 53%. Last, confident normal scans and sleep recordings. It's a simple rule, not a second model, so it's easy to explain and to audit.

**Why have a PDF report?**
Doctors share and file results outside the app: in a referral letter, a patient's file, or a message to a colleague. The PDF is one page with the scan, heatmap, probabilities, the checks and the clinician's review, and it says on the page that it's not a diagnosis. Downloading one is logged like any other access.

**How do you stop one patient seeing another's data?**
Every patient endpoint calls one access check. A patient can open only their own results; a clinician only those of patients who are sharing with them; an admin none. Records you can't see return "not found" rather than "forbidden", so IDs can't be probed. There are automated tests for each of these.

**How fast is it, and what does it cost?**
About {latency} ms per MRI forward pass on a two-core CPU, and a full request with heatmap, uncertainty and stability check takes about {req} seconds. The prototype runs free on Hugging Face; a dedicated 8-vCPU instance is $0.03 an hour.

**What are the main limitations?**
Single 2-D slices rather than 3-D volumes; one public dataset with no information on scanners or patient demographics; no test on Indian hospital data; sleep trained on 20 nights from healthy adults; no clinical validation yet.

## The business

**How would you make money?**
Per-study pricing for small centres (₹{price_mri} per MRI) or a clinic plan of ₹{clinic} a month, and annual licences for hospital chains and teleradiology companies.

**What stops a hospital using this tomorrow?**
Regulation. AI software that informs diagnosis is a medical device in India and needs CDSCO registration, which requires clinical evidence, including how the training and test data were composed and how the model performs across groups. That's why the plan starts with a silent pilot.

**How are you different from Qure.ai?**
Qure.ai is large and established, mainly in chest X-ray and emergency head CT, sold as enterprise contracts. Neurova focuses on MRI tumour triage for small centres at per-study prices, and gives patients control and visibility over who sees their scans.
