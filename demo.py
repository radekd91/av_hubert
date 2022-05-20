
import dlib, cv2, os
import numpy as np
import skvideo
import skvideo.io
from tqdm import tqdm
from avhubert.preparation.align_mouth import landmarks_interpolate, crop_patch, write_video_ffmpeg
from base64 import b64encode
import cv2
import tempfile
from argparse import Namespace
import fairseq
from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.dataclass.configs import GenerationConfig
from avhubert import hubert_pretraining, hubert, hubert_asr
import cv2
import tempfile
import torch
import avhubert.utils as avhubert_utils
from argparse import Namespace
from pathlib import Path
import pickle as pkl
from scipy.io import wavfile
import torch.nn.functional as F
from python_speech_features import logfbank
from scipy.io import wavfile
from copy import deepcopy

def stacker(feats, stack_order):
  """
  Concatenating consecutive audio frames
  Args:
  feats - numpy.ndarray of shape [T, F]
  stack_order - int (number of neighboring frames to concatenate
  Returns:
  feats - numpy.ndarray of shape [T', F']
  """
  feat_dim = feats.shape[1]
  if len(feats) % stack_order != 0:
      res = stack_order - len(feats) % stack_order
      res = np.zeros([res, feat_dim]).astype(feats.dtype)
      feats = np.concatenate([feats, res], axis=0)
  feats = feats.reshape((-1, stack_order, feat_dim)).reshape(-1, stack_order*feat_dim)
  return feats

def play_video(video_path, width=200):
  mp4 = open(video_path,'rb').read()
  data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
  return HTML(f"""
  <video width={width} controls>
        <source src="{data_url}" type="video/mp4">
  </video>
  """)

def detect_landmark(image, detector, predictor):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    rects = detector(gray, 1)
    coords = None
    for (_, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        coords = np.zeros((68, 2), dtype=np.int32)
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def crop_video(input_landmarks_path, input_video_path, output_video_path):
  # load landmarks with pickle 
  with open(input_landmarks_path, 'rb') as f:
    preprocessed_landmarks = pkl.load(f)

  STD_SIZE = (256, 256)
  mean_face_landmarks = np.load(mean_face_path)
  stablePntsIDs = [33, 36, 39, 42, 45]
  rois = crop_patch(input_video_path, preprocessed_landmarks, mean_face_landmarks, stablePntsIDs, STD_SIZE, 
                        window_margin=12, start_idx=48, stop_idx=68, crop_height=96, crop_width=96)
  write_video_ffmpeg(rois, output_video_path, "/usr/bin/ffmpeg")


def preprocess_video(input_video_path, output_video_path, output_landmarks_path, face_predictor_path, mean_face_path):
  detector = dlib.get_frontal_face_detector()
  predictor = dlib.shape_predictor(face_predictor_path)
  STD_SIZE = (256, 256)
  mean_face_landmarks = np.load(mean_face_path)
  stablePntsIDs = [33, 36, 39, 42, 45]
  videogen = skvideo.io.vread(input_video_path)
  frames = np.array([frame for frame in videogen])
  landmarks = []
  for frame in tqdm(frames):
      landmark = detect_landmark(frame, detector, predictor)
      landmarks.append(landmark)
  preprocessed_landmarks = landmarks_interpolate(landmarks)

  # save landmarks using pickle
  with open(output_landmarks_path, 'wb') as f:
    pkl.dump(preprocessed_landmarks, f)

  rois = crop_patch(input_video_path, preprocessed_landmarks, mean_face_landmarks, stablePntsIDs, STD_SIZE, 
                        window_margin=12, start_idx=48, stop_idx=68, crop_height=96, crop_width=96)
  write_video_ffmpeg(rois, output_video_path, "/usr/bin/ffmpeg")
  return


def preprocess_audio(input_audio_path, output_audio_path, sample_rate=16000):
  # extract the audio from the input video using ffmpeg, sample rate = 16000 
  os.system(f"ffmpeg -i {str(input_audio_path)} -ab 160k -ac 1 -ar {sample_rate} {str(output_audio_path)}")
  return

def load_model(ckpt_path):
  models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
  models = [model.eval().cuda() for model in models]
  return models, saved_cfg, task


def predict(video_path, audio_path, models, saved_cfg, task, user_dir, use_video_only):
  num_frames = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))
  data_dir = tempfile.mkdtemp()
  # tsv_cont = ["/\n", f"test-0\t{video_path}\t{None}\t{num_frames}\t{int(16_000*num_frames/25)}\n"]
  if use_video_only:
    tsv_cont = ["/\n", f"test-0\t{video_path}\t{None}\t{num_frames}\t{int(16_000*num_frames/25)}\n"]
  else: 
    tsv_cont = ["/\n", f"test-0\t{video_path}\t{audio_path}\t{num_frames}\t{int(16_000*num_frames/25)}\n"]

  label_cont = ["DUMMY\n"]
  with open(f"{data_dir}/test.tsv", "w") as fo:
    fo.write("".join(tsv_cont))
  with open(f"{data_dir}/test.wrd", "w") as fo:
    fo.write("".join(label_cont))
  utils.import_user_module(Namespace(user_dir=user_dir))
  # modalities = ["video"]
  gen_subset = "test"
  gen_cfg = GenerationConfig(beam=20)
  # models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
  # models = [model.eval().cuda() for model in models]
  # saved_cfg.task.modalities = modalities
  saved_cfg.task.data = data_dir
  saved_cfg.task.label_dir = data_dir
  if 'noise_wav' in saved_cfg.task.keys():
    saved_cfg.task.noise_wav = None # added by me to load the audio-visual model that expect a wav file of background noise when trained (not sure why for inference)
  task = tasks.setup_task(saved_cfg.task)
  task.load_dataset(gen_subset, task_cfg=saved_cfg.task)
  generator = task.build_generator(models, gen_cfg)

  def decode_fn(x):
      dictionary = task.target_dictionary
      symbols_ignore = generator.symbols_to_strip_from_output
      symbols_ignore.add(dictionary.pad())
      return task.datasets[gen_subset].label_processors[0].decode(x, symbols_ignore)

  itr = task.get_batch_iterator(dataset=task.dataset(gen_subset)).next_epoch_itr(shuffle=False)
  sample = next(itr)
  sample = utils.move_to_cuda(sample)
  hypos = task.inference_step(generator, models, sample)
  ref = decode_fn(sample['target'][0].int().cpu())
  hypo = hypos[0][0]['tokens'].int().cpu()
  hypo = decode_fn(hypo)
  return hypo



def extract_feature(video_path, audio_path, models, saved_cfg, task, user_dir, is_finetune_ckpt=False):
  utils.import_user_module(Namespace(user_dir=user_dir))
  # models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
  transform = avhubert_utils.Compose([
      avhubert_utils.Normalize(0.0, 255.0),
      avhubert_utils.CenterCrop((task.cfg.image_crop_size, task.cfg.image_crop_size)),
      avhubert_utils.Normalize(task.cfg.image_mean, task.cfg.image_std)])
  frames = avhubert_utils.load_video(video_path)
  print(f"Load video {video_path}: shape {frames.shape}")
  frames = transform(frames)
  print(f"Center crop video to: {frames.shape}")
  frames = torch.FloatTensor(frames).unsqueeze(dim=0).unsqueeze(dim=0).cuda()

  if 'audio' in saved_cfg.task.modalities:
    sample_rate, wav_data = wavfile.read(audio_path)
    assert sample_rate == 16_000 and len(wav_data.shape) == 1
    # if np.random.rand() < model.noise_prob: # this would be noise augmentation
    #   wav_data = model.add_noise(wav_data)
    audio_feats = logfbank(wav_data, samplerate=sample_rate).astype(np.float32) # [T, F]
    audio_feats = stacker(audio_feats, saved_cfg.task.stack_order_audio) # [T/stack_order_audio, F*stack_order_audio]
    audio_feats = torch.tensor(audio_feats).transpose(0,1).unsqueeze(0).cuda()
    if saved_cfg.task.normalize: 
      with torch.no_grad():
          audio_feats = F.layer_norm(audio_feats, audio_feats.shape[1:])
  else: 
    audio_feats = None

  model = models[0]
  if hasattr(models[0], 'decoder'):
    print(f"Checkpoint: fine-tuned")
    model = models[0].encoder.w2v_model
  else:
    print(f"Checkpoint: pre-trained w/o fine-tuning")
  model.cuda()
  model.eval()
  with torch.no_grad():
    # Specify output_layer if you want to extract feature of an intermediate layer
    feature, _ = model.extract_finetune(source={'video': frames, 'audio': audio_feats}, padding_mask=None, output_layer=None)
    feature = feature.squeeze(dim=0)
  print(f"Video feature shape: {feature.shape}")
  return feature


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def process_video(model_name, model, video_name, input_video_folder, landmarks_from=None, use_video_only=False): 
    f = Path(__file__).absolute().parents[1]
    folder = f / "data"
    landmarks_from = video_name if landmarks_from is None else landmarks_from

    face_predictor_path = f"{folder}/dlib/shape_predictor_68_face_landmarks.dat"
    mean_face_path =  f"{folder}/tcn/20words_mean_face.npy"
    # origin_clip_path =  f"{folder}/input_videos/{video_name}"
    # origin_clip_path =  f"{folder}/input_videos_25fps/{video_name}"
    origin_clip_path =  f"{folder}/{input_video_folder}/{video_name}"
    mouth_roi_path =  f"{folder}/preprocess/roi/{video_name}"
    audio_path = f"{folder}/preprocess/audio/{video_name}"
    audio_path = audio_path.replace('.mp4', '.wav')

    Path(mouth_roi_path).parent.mkdir(parents=True, exist_ok=True)
    landmarks_path =  f"{folder}/preprocess/landmarks/{landmarks_from}"
    Path(landmarks_path).parent.mkdir(parents=True, exist_ok=True)
    if Path(landmarks_from).name == Path(video_name).name: 
        if not Path(mouth_roi_path).is_file() or not Path(landmarks_path).is_file():
            preprocess_video(origin_clip_path, mouth_roi_path, landmarks_path, face_predictor_path, mean_face_path)

    Path(audio_path).parent.mkdir(parents=True, exist_ok=True)
    if not Path(audio_path).is_file(): 
      preprocess_audio(origin_clip_path, audio_path)
    
    if Path(landmarks_from).name != Path(video_name).name and not Path(mouth_roi_path).is_file(): 
        crop_video(landmarks_path, origin_clip_path, mouth_roi_path)

    # play_video(mouth_roi_path)

    # user_dir =  f"{folder}/av_hubert/avhubert"
    user_dir = "."

    models, saved_cfg, task = model
    if 'noise_prob' in saved_cfg.task.keys():
      saved_cfg.task.noise_prob = 0.0
    if use_video_only: 
      saved_cfg.task.modalities = ['video']

    hypo = predict(mouth_roi_path, audio_path, models, deepcopy(saved_cfg), task, user_dir, use_video_only)
    

    # user_dir =  f"{folder}/av_hubert/avhubert"
    feature = extract_feature(mouth_roi_path, audio_path, models, deepcopy(saved_cfg), task, user_dir)

    print("Model encoder:")
    print(count_parameters(models[0].encoder.w2v_model))

    print(f"Prediction:")
    print(hypo)

    prediction_file = Path(f"{folder}/predictions/{video_name}/{model_name}.txt") 
    prediction_file.parent.mkdir(parents=True, exist_ok=True)
    with open(prediction_file, "w") as f:
        f.write(hypo)
  
    feature_file = Path(f"{folder}/visual_features/{video_name}/{model_name}.npy")
    feature_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(feature_file, feature.detach().cpu().numpy())


if __name__ == "__main__":
    # os.chdir(Path(__file__).parent) 
    # folder = "/content"
    print(__file__)
    f = Path(__file__).absolute().parents[1]
    folder = f / "data"
    use_video_only = True
    # model_name = "finetune-model.pt"
    model_name = "vsr/self_large_vox_433h.pt" 
    # model_name = "vsr/self_large_vox_30h.pt" 
    # model_name = "vsr/base_vox_433h.pt"
    # model_name = "avsr/large_noise_pt_noise_ft_30.pt" 
    # model_name = "avsr/large_noise_pt_noise_ft_30h.pt" 
    # model_name = "avsr/large_noise_pt_noise_ft_433h.pt" 
    # model_name = "avsr/base_noise_pt_noise_ft_30h.pt" 
    # model = "avsr/base_noise_pt_noise_ft_433h.pt" 
    checkpoint_folder = f / "checkpoints"
    video_names = []

    sentences = [f"sentence{i:02d}" for i in range(1,2)]
    # sentences = [f"sentence{i:02d}" for i in range(2,11)]
    # sentences = [f"sentence{i:02d}" for i in range(11,12)]

    subjects =  [f.name  for f in list(Path(f"{folder}/input_videos/vocaset/").glob("*")) if f.is_dir()]
    subjects = subjects[0:1]

    # subjects = [P.glo sentence in sentences]

    # video_name = "example.mp4"
    # video_name = "example_emoca_tex_background.mp4"
    # video_name = "example_emoca_tex_nobackground.mp4"
    # video_name = "example_emoca_geo_nobackground.mp4"
    # video_name = "example_emoca_geo_background.mp4"
    # video_name = "sentence29_26_C_with_sound.mp4"
    # video_name = "sentence29_26_C_with_sound_emoca.mp4"
    # video_name = "sentence29_26_C_pytorch3d_with_sound.mp4"
    # video_names += ["sentence01_26_C_with_sound.mp4"]
    # video_names += ["sentence01_26_C_masked_with_sound.mp4"]
    # video_names += ["sentence01_26_C_pytorch3d_with_sound.mp4"]
    # video_names += ["sentence01_26_C_pytorch3d_masked_with_sound.mp4"]
    # video_name = "FaceTalk_170811_03274_TA_sentence01_26_C_with_sound.mp4"
    # video_name = "target_transfer_000.mp4"
    # video_name = "target_transfer_001.mp4"

    # landmarks_from = "example.pkl"
    # landmarks_from = "sentence29_26_C_with_sound.mp4"

    input_video_folder = "input_videos"
    # input_video_folder = "input_videos_25fps"
    
    video_names = [] 
    for subject in subjects:
      for sentence in sentences:
          video_names += [f"vocaset/{subject}/{sentence}/{sentence}_26_C_with_sound.mp4"]
          video_names += [f"vocaset/{subject}/{sentence}/{sentence}_26_C_masked_with_sound.mp4"]
          video_names += [f"vocaset/{subject}/{sentence}/{sentence}_26_C_pytorch3d_with_sound.mp4"]
          video_names += [f"vocaset/{subject}/{sentence}/{sentence}_26_C_pytorch3d_masked_with_sound.mp4"]
          
    ckpt_path =  f"{checkpoint_folder}/{model_name}"
    model =  load_model(ckpt_path)

    for video_name in video_names: 
      landmarks_from = video_name
      process_video(model_name, model, video_name, input_video_folder , landmarks_from=landmarks_from, use_video_only=use_video_only)


    # face_predictor_path = f"{folder}/dlib/shape_predictor_68_face_landmarks.dat"
    # mean_face_path =  f"{folder}/tcn/20words_mean_face.npy"
    # # origin_clip_path =  f"{folder}/input_videos/{video_name}"
    # origin_clip_path =  f"{folder}/input_videos_25fps/{video_name}"
    # mouth_roi_path =  f"{folder}/preprocess/roi/{video_name}"
    # landmarks_path =  f"{folder}/preprocess/landmarks/{landmarks_from}"
    # if Path(landmarks_from).name == Path(video_name).name: 
    #     if not Path(mouth_roi_path).is_file() or not Path(landmarks_path).is_file():
    #         preprocess_video(origin_clip_path, mouth_roi_path, landmarks_path, face_predictor_path, mean_face_path)
    
    # if Path(landmarks_from).name != Path(video_name).name and not Path(mouth_roi_path).is_file(): 
    #     crop_video(landmarks_path, origin_clip_path, mouth_roi_path)

    # # play_video(mouth_roi_path)

    # ckpt_path =  f"{checkpoint_folder}/{model_name}"
    # # user_dir =  f"{folder}/av_hubert/avhubert"
    # user_dir = "."

    # models, saved_cfg, task = load_model(ckpt_path)

    # hypo = predict(mouth_roi_path, models, saved_cfg, task, user_dir)
    

    # # user_dir =  f"{folder}/av_hubert/avhubert"
    # feature = extract_visual_feature(mouth_roi_path, models, saved_cfg, task, user_dir)

    # print("Model encoder:")
    # print(count_parameters(models[0].encoder.w2v_model))

    # print(f"Prediction:")
    # print(hypo)

    # prediction_file = f"{folder}/predictions/{video_name}.txt" 
    # with open(prediction_file, "w") as f:
    #     f.write(hypo)
  
    # feature_file = f"{folder}/features/{video_name}.npy"
    # np.save(feature_file, feature.detach().cpu().numpy())



