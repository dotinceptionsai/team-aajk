from pathlib import Path

import mlflow
import pandas as pd
import streamlit as st

from analysis import experiments, evaluate
from backoffice import shared
from backoffice.session import SessionKey
from backoffice.shared import make_sidebar
from pipelines import persistence

st.session_state.update(
    {k: v for k, v in st.session_state.items() if k.startswith("SessionKey")}
)
datasets = shared.get_ds_registry()

shared.make_header(3)
st.title("Model Calibration 🪄")

st.write(
    "Choose between a faster model that is less accurate or a slower one but more accurate."
)

make_sidebar()

if SessionKey.SELECTED_DATASET not in st.session_state:
    shared.go_back_to("Data Selection", "Select dataset first")
elif SessionKey.SELECTED_BASE_MODEL not in st.session_state:
    shared.go_back_to("Model Choice", "Select base model first")
else:

    ds = st.session_state[SessionKey.SELECTED_DATASET]
    base_model_name = st.session_state[SessionKey.SELECTED_BASE_MODEL]

    mlruns = str(str(Path("train/mlruns").absolute()))
    mlflow.set_tracking_uri("file://" + mlruns)
    run_info = experiments.get_run_info(experiment_name=ds, run_name=base_model_name)

    pipeline_dir = Path(run_info.artifact_dir.replace("file:/", "/"))
    st.write("Pipeline dir exists: ", pipeline_dir.exists())
    st.write("Pipeline dir: ", pipeline_dir)
    model = persistence.load_pipeline(pipeline_dir)

    with st.expander("Base model details"):
        st.write(persistence._create_model_dict(model))

    from pipelines.impl.anomaly_detection import OnInvalidSentence

    default_id_sentences = datasets.load_items(model.datasets.validation_id)
    default_ood_sentences = datasets.load_items(model.datasets.validation_ood)

    id_sentences = None

    with st.form("form"):
        st.caption(
            "Enter your sentences to help the model understand the limits of what is ID and OOD"
        )
        st.caption(
            "Try to enter sentences that are just at the limit of being out-of-domain or in-domain."
        )

        with st.expander("Expand to input your in-domain sentences"):
            st.caption(
                "Try to enter sentences that are just at the limit of being in-domain."
            )
            id_sentences_ta = st.text_area(
                "ID sentences",
                value="\n".join(default_id_sentences),
                help="Enter some In-Domain sentences here.",
                height=200,
            )

        with st.expander("Expand to input your out-of-domain sentences"):
            st.caption(
                "Try to enter sentences that are just at the limit of being out-of-domain."
            )
            ood_sentences_ta = st.text_area(
                "OOD sentences",
                value="\n".join(default_ood_sentences),
                help="Enter some Out-of-Domain sentences here.",
                height=200,
            )

        recalibration_btn = st.form_submit_button("Refresh distances and metrics")

        if recalibration_btn:
            with st.spinner("Recalibrating model..."):
                id_sentences = id_sentences_ta.splitlines()
                ood_sentences = ood_sentences_ta.splitlines()
                model.recalibrate(
                    id_sentences,
                    ood_sentences,
                    registry=None,
                    on_invalid_sentence=OnInvalidSentence.WARN,
                )
                st.session_state.manual_cutoff = model.calibrator.cutoff_
                st.write(
                    "Cutoff distance that maximizes F1: ",
                    round(model.calibrator.cutoff_, 2),
                )

                adjustment = st.slider(
                    "Adjust cutoff distance for more precision or recall. Click button again to refresh metrics.",
                    model.calibrator.r_id_,
                    model.calibrator.r_ood_,
                    model.calibrator.cutoff_,
                )

                with st.expander("Sentence distances to center of In-Domain (ID)"):
                    st.caption(
                        "Every sentence with a distance below this cutoff will be considered ID, above will be considered OOD."
                    )

                    raw_id_scores = model.train_pipe.transform(id_sentences)
                    raw_ood_scores = model.train_pipe.transform(ood_sentences)

                    df_id = pd.DataFrame({"distance": raw_id_scores, "origin": "ID"})
                    df_ood = pd.DataFrame({"distance": raw_ood_scores, "origin": "OOD"})
                    df = pd.concat([df_id, df_ood])
                    df["sentence"] = [*id_sentences, *ood_sentences]

                    df = df.sort_values(by="distance", ascending=True)

                    import seaborn as sns
                    import matplotlib.pyplot as plt

                    sns.set_style("darkgrid")
                    fig, ax = plt.subplots()
                    graph = sns.histplot(
                        data=df,
                        x="distance",
                        hue="origin",
                        legend=True,
                        stat="count",
                        palette=["red", "lightblue"],
                        alpha=0.5,
                        kde=True,
                        ax=ax,
                    )
                    graph.axvline(
                        model.calibrator.cutoff_,
                        color="grey",
                        linestyle="--",
                        label="Cutoff",
                    )
                    st.caption(
                        "Histogram shows sentences distance distribution. Check :blue[ID] and :red[OOD] are well separated."
                    )
                    st.pyplot(fig)

                    st.caption(
                        "Table shows each sentences distance to center of the ID. It helps you see where you want the cutoff to be."
                    )
                    st.dataframe(
                        df.style.format(formatter={"distance": "{:.2f}"}),
                        use_container_width=True,
                    )

                with st.expander(expanded=True, label="Evaluation metrics"):
                    manual_cutoff = st.session_state["manual_cutoff"]
                    metrics = evaluate.evaluate_model(
                        model, id_sentences, ood_sentences
                    )
                    f1 = round(float(metrics.f1), 2)
                    st.write("f1", f1)

                    fp_sentences = pd.DataFrame(
                        {"sentence": [id_sentences[i] for i in metrics.fp_indices]}
                    )
                    fn_sentences = pd.DataFrame(
                        {"sentence": [ood_sentences[i] for i in metrics.fn_indices]}
                    )

                    st.write("False positives")
                    st.write(fp_sentences)

                    st.write("False negatives")
                    st.write(fn_sentences)

    # cm = confusion_matrix(metrics.y_true, metrics.y_pred)
    # ConfusionMatrixDisplay(cm).plot()
    # plt.savefig(output_dir / "confusion_matrix.png")
    #
    # pred, sent = pipeline.predict_proba(
    #     [*val_id_sentences, *val_ood_sentences])
    # pd.DataFrame({"sentence": sent, "OOD prob": pred}).to_csv(
    #     output_dir / "predictions.csv", index=False
    # )

    # import seaborn as sns
    # import matplotlib.pyplot as plt
    #
    # sns.set_style("darkgrid")
    # fig, ax = plt.subplots()
    # graph = sns.histplot(
    #     data=df,
    #     x="score",
    #     hue="origin",
    #     legend=True,
    #     stat="count",
    #     palette=["red", "lightblue"],
    #     alpha=0.5,
    #     kde=True,
    #     ax=ax,
    # )
    # graph.axvline(
    #     model.calibrator.cutoff_, color="grey", linestyle="--",
    #     label="Cutoff"
    # )
    # st.pyplot(fig)
