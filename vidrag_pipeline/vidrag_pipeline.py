from PIL import Image
import torch
print(torch.cuda.is_available()) 
print(torch.version.cuda)  
# from llava.model.builder import load_pretrained_model
# from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
# from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
# from llava.conversation import conv_templates, SeparatorStyle
import torch
from transformers import AutoProcessor, WhisperForConditionalGeneration, WhisperProcessor, CLIPProcessor, CLIPModel
import copy
from decord import VideoReader, cpu
import ffmpeg, torchaudio
from openai import OpenAI
import easyocr

import os
import numpy as np
import json
from tqdm import tqdm
import os
from tools.rag_retriever_dynamic import retrieve_documents_with_dynamic
import re
import ast
import socket
import pickle
from tools.filter_keywords import filter_keywords
from tools.scene_graph import generate_scene_graph_description

       

max_frames_num = 32
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336", torch_dtype=torch.float16, device_map="auto")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
whisper_model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-large",
    torch_dtype=torch.float16,
    device_map="auto"
)
whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-large")

def process_video(video_path, max_frames_num, fps=1, force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(),num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps()/fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i/fps for i in frame_idx]
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()

    return spare_frames, frame_time, video_time

def extract_audio(video_path, audio_path):
    if not os.path.exists(audio_path):
        ffmpeg.input(video_path).output(audio_path, acodec='pcm_s16le', ac=1, ar='16k').run()

def chunk_audio(audio_path, chunk_length_s=30):
    speech, sr = torchaudio.load(audio_path)
    speech = speech.mean(dim=0)  
    speech = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(speech)  
    num_samples_per_chunk = chunk_length_s * 16000 
    chunks = []
    for i in range(0, len(speech), num_samples_per_chunk):
        chunks.append(speech[i:i + num_samples_per_chunk])
    return chunks

def transcribe_chunk(chunk):

    # inputs = whisper_processor(chunk, return_tensors="pt")
    # inputs["input_features"] = inputs["input_features"].to(whisper_model.device, torch.float16)
    # with torch.no_grad():
    #     predicted_ids = whisper_model.generate(
    #         inputs["input_features"],
    #         no_repeat_ngram_size=2,
    #         early_stopping=True
    #     )
    # transcription = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    client = OpenAI(
        api_key=os.getenv("API_KEY"),
        base_url="https://llm-proxy.imla.hs-offenburg.de/" 
    )
    
    transcription = client.audio.transcriptions.create(
        model="whisper-1", 
        file=chunk, 
        response_format="text"
        # timestamp_granularities=["segment"] 
    )
    return transcription
    

def get_asr_docs(video_path, audio_path):

    full_transcription = []

    try:
        extract_audio(video_path, audio_path)
    except:
        return full_transcription
    audio_chunks = chunk_audio(audio_path, chunk_length_s=30)
    
    for chunk in audio_chunks:
        transcription = transcribe_chunk(chunk)
        full_transcription.append(transcription)

    return full_transcription

# def get_ocr_docs(frames):
#     reader = easyocr.Reader(['en']) 
#     text_set = []
#     ocr_docs = []
#     for img in frames:
#         ocr_results = reader.readtext(img)
#         det_info = ""
#         for result in ocr_results:
#             text = result[1]
#             confidence = result[2]
#             if confidence > 0.5 and text not in text_set:
#                 det_info += f"{text}; "
#                 text_set.append(text)
#         if len(det_info) > 0:
#             ocr_docs.append(det_info)

#     return ocr_docs


def get_ocr_docs(frames):
    ocr_results = []

    for img in frames:
        try:
            image_bytes = base64.b64decode(img)
            image = Image.open(BytesIO(image_bytes)).convert("RGB")

            text = pytesseract.image_to_string(image, lang='eng')        
            ocr_results.append(text)
        except Exception as e:
            print(f"⚠️ OCR failed: {e}")

    return ocr_results

def save_frames(frames):
    file_paths = []
    for i, frame in enumerate(frames):
        img = Image.fromarray(frame)
        file_path = f'restore/frame_{i}.png'
        img.save(file_path)
        file_paths.append(file_path)
    return file_paths
    
def get_det_docs(frames, prompt):
    prompt = ",".join(prompt)
    frames_path = save_frames(frames)
    res = []
    if len(frames) > 0:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect(('0.0.0.0', 9999))
        data = (frames_path, prompt)
        client_socket.send(pickle.dumps(data))
        result_data = client_socket.recv(4096)
        try:
            res = pickle.loads(result_data)
        except:
            res = []
    return res

def det_preprocess(det_docs, location, relation, number):

    scene_descriptions = []

    for det_doc_per_frame in det_docs:
        objects = []
        scene_description = ""
        if len(det_doc_per_frame) > 0:
            for obj_id, objs in enumerate(det_doc_per_frame.split(";")):
                obj_name = objs.split(":")[0].strip()
                obj_bbox = objs.split(":")[1].strip()
                obj_bbox = ast.literal_eval(obj_bbox)
                objects.append({"id": obj_id, "label": obj_name, "bbox": obj_bbox})

            scene_description = generate_scene_graph_description(objects, location, relation, number)
        scene_descriptions.append(scene_description)
    
    return scene_descriptions


# # load your VLM
# device = "cuda"
# overwrite_config = {}
# tokenizer, model, image_processor, max_length = load_pretrained_model(
#     "LLaVA-Video-7B-Qwen2", 
#     None, 
#     "llava_qwen", 
#     torch_dtype="bfloat16", 
#     device_map="auto", 
#     overwrite_config=overwrite_config)  # Add any other thing you want to pass in llava_model_args
# model.eval()
# conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models


# The inference function of your VLM
# def llava_inference(qs, video):
#     if video is not None:
#         question = DEFAULT_IMAGE_TOKEN + qs
#     else:
#         question = qs
#     conv = copy.deepcopy(conv_templates[conv_template])
#     conv.append_message(conv.roles[0], question)
#     conv.append_message(conv.roles[1], None)
#     prompt_question = conv.get_prompt()
#     input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
#     cont = model.generate(
#         input_ids,
#         images=video,
#         modalities= ["video"],
#         do_sample=False,
#         temperature=0,
#         max_new_tokens=4096,
#     )
#     text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
#     return text_outputs

import openai
from PIL import Image
import base64
from io import BytesIO

# Helper function: encode a single image (e.g., frame)
def encode_image(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Inference function for GPT-4o with image input
def gpt4o_inference(question, frames):
    MODEL="gpt-4o-mini"
    client = openai.OpenAI(api_key=os.getenv("API_KEY"),
                           base_url="https://llm-proxy.imla.hs-offenburg.de")
    
    response =  client.chat.completions.create(
            model=MODEL,
            messages = [
                {"role": "system", "content": "You are a helpful assistant that answers questions based on video frames."},
                    # User's question (similar to what LLaVA appends)
                    {"role": "user", "content": question},
                    *[
                        {"role": "user", "content": f"data:image/png;base64,{encode_image(frame)}"}
                        for frame in frames
                    ]
                    ], 
                    temperature=0
                )
            
    
    return response['choices'][0]['message']['content']


# super-parameters setting
rag_threshold = 0.3
clip_threshold = 0.3
beta = 3.0 

# Choose the auxiliary texts you want
USE_OCR = True
USE_ASR = True
USE_DET = True
print(f"---------------OCR {rag_threshold}: {USE_OCR}-----------------")
print(f"---------------ASR {rag_threshold}: {USE_ASR}-----------------")
print(f"---------------DET {beta}-{clip_threshold}: {USE_DET}-----------------")
print(f"---------------Frames: {max_frames_num}-----------------")


video_path = "/home/jos/VidRagTool/data/terminal_bedienung.mp4"  # your video path
question = "Wie identifiziert sich der User?"  # your question


frames, frame_time, video_time = process_video(video_path, max_frames_num, 1, force_sample=True)
raw_video = [f for f in frames]
from PIL import Image

# Convert (N, H, W, C) NumPy array to list of PIL images
frame_list = [Image.fromarray(frame.astype("numpy.float32")).convert("RGB") for frame in frames]

# video = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].cuda().bfloat16()
def decode_base64_image(b64_string):
    image_data = base64.b64decode(b64_string)
    image = Image.open(BytesIO(image_data)).convert("RGB")  # Ensure 3 channels
    return image

# Decode all your 64 base64 frames to PIL images
# decoded_frames = [decode_base64_image(f) for f in frames] 
# video = [video]

if USE_DET:
    video_tensor = []
    for frame in raw_video:
        processed = clip_processor(images=frame, return_tensors="pt")["pixel_values"].to(clip_model.device, dtype=torch.float16)
        video_tensor.append(processed.squeeze(0))
    video_tensor = torch.stack(video_tensor, dim=0)

if USE_OCR:
    ocr_docs_total = get_ocr_docs(frames)

if USE_ASR:
    if os.path.exists(os.path.join("restore/audio", os.path.basename(video_path).split(".")[0] + ".txt")):
        with open(os.path.join("restore/audio", os.path.basename(video_path).split(".")[0] + ".txt"), 'r', encoding='utf-8') as f:
            asr_docs_total = f.readlines()
    else:
        audio_path = os.path.join("restore/audio", os.path.basename(video_path).split(".")[0] + ".wav")
        asr_docs_total = get_asr_docs(video_path, audio_path)
        with open(os.path.join("restore/audio", os.path.basename(video_path).split(".")[0] + ".txt"), 'w', encoding='utf-8') as f:
            for doc in asr_docs_total:
                f.write(doc + '\n')

# step 0: get cot information
retrieve_pmt_0 = "Frage: " + question
# you can change this decouple prompt to fit your requirements
retrieve_pmt_0 += "\nUm die Frage Schritt für Schritt zu beantworten, können Sie Ihre Abrufanfrage im folgenden json-Format bereitstellen:"
retrieve_pmt_0 += '''{
    "ASR": Optional[str]. Die Untertitel des Videos, die für die Frage, die Sie beantworten möchten, relevant sein könnten, in zwei Sätzen. Wenn Sie diese Informationen nicht benötigen, geben Sie bitte null zurück.
    "DET": Optional[list]. (Die Ausgabe darf nur physische Entitäten enthalten, keine abstrakten Konzepte, weniger als fünf Entitäten) Alle physischen Entitäten und ihr Standort im Zusammenhang mit der Frage, die Sie abrufen möchten, keine abstrakten Konzepte. Wenn Sie diese Informationen nicht benötigen, geben Sie bitte null zurück.
    "TYPE": Optional[list]. (Die Ausgabe muss als Null oder als Liste angegeben werden, die nur eine oder mehrere der folgenden Zeichenketten enthält: 'Ort', 'Nummer', 'Beziehung'. Andere Werte sind für dieses Feld nicht zulässig) Die Informationen, die Sie über die erkannten Objekte erhalten möchten. Wenn Sie die Position des Objekts im Videobild benötigen, geben Sie "location" aus; wenn Sie die Nummer eines bestimmten Objekts benötigen, geben Sie "number" aus; wenn Sie die Positionsbeziehung zwischen Objekten benötigen, geben Sie "relation" aus.
}
## Beispiel 1:
Frage: Wie viele blaue Luftballons befinden sich über dem langen Tisch in der Mitte des Raumes am Ende des Videos? A. 1. B. 2. C. 3. D. 4.
Ihr Abruf kann sein:
{
    "ASR": "Der Ort und die Farbe der Luftballons, die Anzahl der blauen Luftballons.",
    "DET": ["blaue Luftballons", "langer Tisch"],
    "TYP": ["relation", "number"]
}
## Beispiel 2:
Frage: Welche Farbe trägt die Frau in der linken unteren Ecke des Videos rechts neben dem Mann in schwarzer Kleidung? A. Blau. B. Weiß. C. Rot. D. Gelb.
Ihr Abruf kann sein:
{
    "ASR": null,
    "DET": ["der Mann in Schwarz", "Frau"],
    "TYP": ["Ort", "Beziehung"]
}
## Beispiel 3:
Frage: In welchem Land ist die im Video gezeigte Komödie weltweit bekannt? A. China. B. GROSSBRITANNIEN. C. Deutschland. D. Vereinigte Staaten.
Ihr Abruf kann sein:
{
    "ASR": "Das Land, das weltweit für seine Komödie bekannt ist.",
    "DET": null,
    "TYP": null
}
Beachten Sie, dass Sie die Frage in diesem Schritt nicht beantworten müssen, also brauchen Sie keine Informationen über das Video oder das Bild. Sie müssen nur Ihre Abrufanfrage angeben (optional), und ich werde Ihnen helfen, die gewünschten Informationen abzurufen. Bitte geben Sie das json-Format an.'''

json_request, _ = gpt4o_inference(retrieve_pmt_0, None)

# step 1: get docs information
query = [question]

# APE fetch
if USE_DET:
    det_docs = []
    try:
        request_det = json.loads(json_request)["DET"]
        request_det = filter_keywords(request_det)
        clip_text = ["A picture of " + txt for txt in request_det]
        if len(clip_text) == 0:
            clip_text = ["A picture of object"]
    except:
        request_det = None
        clip_text = ["A picture of object"]

    clip_inputs = clip_processor(text=clip_text, return_tensors="pt", padding=True, truncation=True).to(clip_model.device)
    clip_img_feats = clip_model.get_image_features(video_tensor)
    with torch.no_grad():
        text_features = clip_model.get_text_features(**clip_inputs)
        similarities = (clip_img_feats @ text_features.T).squeeze(0).mean(1).cpu()
        similarities = np.array(similarities, dtype=np.float64)
        alpha = beta * (len(similarities) / 16)
        similarities = similarities * alpha / np.sum(similarities)

    del clip_inputs, clip_img_feats, text_features
    torch.cuda.empty_cache()

    det_top_idx = [idx for idx in range(max_frames_num) if similarities[idx] > clip_threshold]
        
    if request_det is not None and len(request_det) > 0:
        det_docs = get_det_docs(frames[det_top_idx], request_det)  

        L, R, N = False, False, False
        try:
            det_retrieve_info = json.loads(json_request)["TYPE"]
        except:
            det_retrieve_info = None
        if det_retrieve_info is not None:
            if "location" in det_retrieve_info:
                L = True
            if "relation" in det_retrieve_info:
                R = True
            if "number" in det_retrieve_info:
                N = True
        det_docs = det_preprocess(det_docs, location=L, relation=R, number=N)  # pre-process of APE information


# OCR fetch
if USE_OCR:
    try:
        request_det = json.loads(json_request)["DET"]
        request_det = filter_keywords(request_det)
    except:
        request_det = None
    ocr_docs = []
    if len(ocr_docs_total) > 0:
        ocr_query = query.copy()
        if request_det is not None and len(request_det) > 0:
            ocr_query.extend(request_det)
        ocr_docs, _ = retrieve_documents_with_dynamic(ocr_docs_total, ocr_query, threshold=rag_threshold)

# ASR fetch
if USE_ASR:
    asr_docs = []
    try:
        request_asr = json.loads(json_request)["ASR"]
    except:
        request_asr = None
    if len(asr_docs_total) > 0:
        asr_query = query.copy()
        if request_asr is not None:
            asr_query.append(request_asr)
        asr_docs, _ = retrieve_documents_with_dynamic(asr_docs_total, asr_query, threshold=rag_threshold)

qs = ""
if USE_DET and len(det_docs) > 0:
    for i, info in enumerate(det_docs):
        if len(info) > 0:
            qs += f"Frame {str(det_top_idx[i]+1)}: " + info + "\n"
    if len(qs) > 0:
        qs = f"\nVideo have {str(max_frames_num)} frames in total, the detected objects' information in specific frames: " + qs
if USE_ASR and len(asr_docs) > 0:
    qs += "\nVideo Automatic Speech Recognition information (given in chronological order of the video): " + " ".join(asr_docs)
if USE_OCR and len(ocr_docs) > 0:
    qs += "\nVideo OCR information (given in chronological order of the video): " + "; ".join(ocr_docs)
qs += "Select the best answer to the following multiple-choice question based on the video and the information (if given). Respond with only the letter (A, B, C, or D) of the correct option. Question: " + question  # you can change this prompt

res = gpt4o_inference(qs, frames_list)
print(res)
