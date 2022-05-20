from pathlib import Path
import numpy as np
from sklearn.manifold import TSNE




def main():
    f = Path(__file__).absolute().parents[1]
    folder = f / "data"
    model_name = "vsr/self_large_vox_433h.pt" 

    # sentences = [f"sentence{i:02d}" for i in range(1,2)]
    sentences = [f"sentence{i:02d}" for i in range(1,10)]

    subjects =  [f.name  for f in list(Path(f"{folder}/input_videos/vocaset/").glob("*")) if f.is_dir()]
    # subjects = [subjects[0]]
    subjects = subjects[0:1]
    # subjects = subjects[0:4]

    # video_names = []
    # video_names += ["sentence01_26_C_with_sound.mp4"]
    # video_names += ["sentence01_26_C_masked_with_sound.mp4"]
    # video_names += ["sentence01_26_C_pytorch3d_with_sound.mp4"]
    # video_names += ["sentence01_26_C_pytorch3d_masked_with_sound.mp4"]
    video_names = [] 
    for subject in subjects:
      for sentence in sentences:
          video_names += [f"vocaset/{subject}/{sentence}/{sentence}_26_C_with_sound.mp4"]
          video_names += [f"vocaset/{subject}/{sentence}/{sentence}_26_C_masked_with_sound.mp4"]
          video_names += [f"vocaset/{subject}/{sentence}/{sentence}_26_C_pytorch3d_with_sound.mp4"]
          video_names += [f"vocaset/{subject}/{sentence}/{sentence}_26_C_pytorch3d_masked_with_sound.mp4"]
    
    features = {}
    predictions = {}

    for video_name in video_names:
        prediction_file = Path(f"{folder}/predictions/{video_name}/{model_name}.txt") 
        prediction_file.parent.mkdir(parents=True, exist_ok=True)
        with open(prediction_file, "r") as f:
            prediction = f.readline()
    
        feature_file = Path(f"{folder}/visual_features/{video_name}/{model_name}.npy")
        feature_file.parent.mkdir(parents=True, exist_ok=True)
        feature = np.load(feature_file)

        features[video_name] = feature
        predictions[video_name] = prediction


    # diffs = {}
    # diffs_magnitude = {}
    # diffs_from = "sentence01_26_C_with_sound.mp4"
    # magnitudes = {}
    # for key in features.keys():
    #     magnitudes[key] = np.linalg.norm(features[key], axis=1)
    #     if key == diffs_from:
    #         continue
    #     diffs[key] = features[diffs_from] - features[key]
    #     diffs_magnitude[key] = np.linalg.norm(diffs[key], axis=1)

    # # plot the maginutes with plotly 
    # import plotly.express as px
    # import plotly.graph_objects as go
    # import pandas as pd
    # # fig = go.Figure()
    # df = pd.DataFrame(diffs_magnitude)
    # fig = px.line(df,# x="t", y="distance", 
    #     title="Distance between sentence01_26_C_with_sound.mp4 and others") 
    # fig.show()


    # df = pd.DataFrame(magnitudes)
    # fig2 = px.line(df,# x="t", y="distance", 
    #     title="Magnitude of the features") 
    # fig2.show()

    # for key in diffs_magnitude.keys():
    #     # px.line(diffs_magnitude[key], x=np.arange(len(diffs_magnitude[key])), title=key)
    #     fig.line(diffs_magnitude[key], x=np.arange(len(diffs_magnitude[key])), title=key)
    #     # fig.add_trace(go.Scatter(x=np.arange(len(diffs_magnitude[key])), y=diffs_magnitude[key], name=key))

    # fig.show()

    all_features = np.concatenate([features[key] for key in features.keys()]) 
    labels = []
    labels_str = []
    subject_str = []
    sentence_str = []
    mode_str = []
    for i, key in enumerate(features.keys()):
        labels += [i] * len(features[key])
        labels_str += [Path(key).name] * len(features[key])
        sentence_str += [Path(key).parents[0].name] * len(features[key])
        subject_str += [Path(key).parents[1].name] * len(features[key])
        mode = Path(key).name[len(f"{sentence}_26_C_"):]
        if mode == "with_sound.mp4":
            mode = "original"
        elif mode == "masked_with_sound.mp4":
            mode = "original masked"
        elif mode == "pytorch3d_with_sound.mp4":
            mode = "pytorch3d"
        elif mode == "pytorch3d_masked_with_sound.mp4":
            mode = "pytorch3d masked"

        mode_str += [mode] * len(features[key])


    X_embedded_space = TSNE(n_components=2, learning_rate='auto',
                  init='random').fit_transform(all_features)

    # create a scatter plot with plotly, color points by labels, add legend 
    import plotly.express as px
    import plotly.graph_objects as go
    import pandas as pd
    df = pd.DataFrame({"x" : X_embedded_space[:,0], "y" : X_embedded_space[:,1], 
                    "labels" : labels, "names" : labels_str, "subjects" : subject_str, "mode" : mode_str, "sentences" : sentence_str})
    fig = px.scatter(df, x="x", y="y", 
                    # color="names", 
                    color="subjects", 
                    # symbol="subjects",
                    # symbol="names",
                    symbol="sentences",
                    labels={
                            "labels": "names", 
                            "subjects": "subjects", 
                            "mode": "mode", 
                            "sentences": "sentences"
                        },
                    title="Visual features in 2D") 
    # plot line segments into the same figure 
    # for i in range(4):
        # fig.add_trace(go.Scatter(x=[df["x"][i], df["x"][i]], y=[df["y"][i], df["y"][i]], mode="lines", name=df["names"][i]))
    # add legend 
    fig.show()


    print("Done")

if __name__ == '__main__':
    main()
