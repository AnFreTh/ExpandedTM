import umap.umap_ as umap
import plotly.express as px
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


from scipy.spatial.distance import cosine


def calculate_distances_to_other_topics(selected_topic_index, plot_df, model_output):
    selected_topic_position = plot_df.iloc[selected_topic_index][["x", "y"]]
    distances = []

    for index, row in plot_df.iterrows():
        if index != selected_topic_index:
            other_topic_position = row[["x", "y"]]
            distance = cosine(
                selected_topic_position, other_topic_position
            )  # Cosine distance
            top_words = ", ".join(
                [word for word, _ in model_output["topic_dict"][index][:3]]
            )  # Top 3 words
            distances.append((index, distance, top_words))

    # Sort by distance
    distances.sort(key=lambda x: x[1])
    return distances


def _visualize_topic_model_2d(model, reduce_first=False, reducer="umap", port=8050):
    num_docs_per_topic = pd.Series(model.labels).value_counts().sort_index()

    # Extract top words for each topic with importance and format them vertically
    top_words_per_topic = {
        topic: "<br>".join(
            [f"{word} ({importance:.2f})" for word, importance in words[:5]]
        )
        for topic, words in model.output["topic_dict"].items()
    }

    if reducer == "umap":
        reducer = umap.UMAP(n_components=2)
    elif reducer == "tsne":
        reducer = TSNE(n_components=2, perplexity=30, learning_rate=200)
    elif reducer == "pca":
        reducer = PCA(n_components=2)
    else:
        raise ValueError("reducer must be in ['umap', 'tnse', 'pca']")

    if reduce_first:
        embeddings = reducer.fit_transform(model.embeddings)
        topic_data = []
        # Iterate over unique labels and compute mean embedding for each
        for label in np.unique(model.labels):
            # Find embeddings corresponding to the current label
            label_embeddings = embeddings[model.labels == label]
            # Compute mean embedding for the current label
            mean_embedding = np.mean(label_embeddings, axis=0)
            # Store the mean embedding in the dictionary
            topic_data.append(mean_embedding)
    else:
        if hasattr(model, "topic_centroids"):
            topic_data = model.topic_centroids
        else:
            topic_data = []
            # Iterate over unique labels and compute mean embedding for each
            for label in np.unique(model.labels):
                # Find embeddings corresponding to the current label
                label_embeddings = model.embeddings[model.labels == label]
                # Compute mean embedding for the current label
                mean_embedding = np.mean(label_embeddings, axis=0)
                # Store the mean embedding in the dictionary
                topic_data.append(mean_embedding)

    topic_embeddings_2d = reducer.fit_transform(topic_data)

    # Create DataFrame for plotting
    plot_df = pd.DataFrame(topic_embeddings_2d, columns=["x", "y"])
    plot_df["topic"] = list(top_words_per_topic.keys())
    plot_df["num_docs"] = num_docs_per_topic
    plot_df["top_words"] = list(top_words_per_topic.values())

    app = dash.Dash(__name__)

    app.layout = html.Div(
        [
            html.H1("Lavish Topic Visualization"),
            dcc.Dropdown(
                id="side-dropdown",
                options=[
                    {"label": "Top Words", "value": "top_words"},
                    {"label": "Topic Distances", "value": "topic_distances"},
                ],
                value="top_words",
            ),
            dcc.Graph(id="main-plot"),
            dcc.Graph(id="side-plot"),
            html.Div(id="details-panel", children="Click a point for details"),
        ]
    )

    @app.callback(Output("main-plot", "figure"), [Input("side-dropdown", "value")])
    def update_main_plot(selected_option):
        fig = px.scatter(
            plot_df,
            x="x",
            y="y",
            color="topic",
            size="num_docs",
            title="Main Topic Plot",
        )
        fig.update_layout(clickmode="event+select")
        return fig

    def get_top_words_for_topic(topic_number, model_output):
        return model_output["topic_dict"].get(topic_number, [])

    @app.callback(
        Output("side-plot", "figure"),
        [Input("main-plot", "clickData"), Input("side-dropdown", "value")],
    )
    def update_side_plot(clickData, side_option):
        if clickData:
            point_index = clickData["points"][0]["pointIndex"]
            selected_topic = point_index

            if side_option == "top_words":
                top_words_data = get_top_words_for_topic(selected_topic, model.output)
                words, scores = zip(*top_words_data)
                fig = px.bar(
                    x=words, y=scores, title=f"Top Words for Topic {selected_topic}"
                )
                return fig

            elif side_option == "topic_distances":
                distances_data = calculate_distances_to_other_topics(
                    point_index, plot_df, model.output
                )
                topics, distances, annotations = zip(*distances_data)
                fig = px.bar(
                    x=topics,
                    y=distances,
                    title=f"Distances from Topic {point_index} to Others",
                )
                # Add annotations
                fig.update_layout(
                    annotations=[
                        dict(
                            x=topic,
                            y=distance,
                            text=annotation,
                            showarrow=True,
                            arrowhead=1,
                            ax=0,
                            ay=-40,
                        )
                        for topic, distance, annotation in zip(
                            topics, distances, annotations
                        )
                    ]
                )

                fig.update_traces(
                    hovertemplate="<br>".join(
                        [
                            "Topic ID: %{x}",
                            "Distance: %{y:.2f}",
                            "<extra></extra>",  # Hides the trace name
                        ]
                    )
                )
                return fig

        return go.Figure()

    @app.callback(
        Output("details-panel", "children"), [Input("main-plot", "clickData")]
    )
    def display_details(clickData):
        if clickData:
            point_index = clickData["points"][0]["pointIndex"]
            selected_topic = point_index
            top_words_data = get_top_words_for_topic(selected_topic, model.output)
            detailed_info = "Topic {}: ".format(selected_topic) + ", ".join(
                [f"{word} ({score:.2f})" for word, score in top_words_data]
            )
            return detailed_info
        return "Click a point for details"

    app.run_server(debug=True, port=port)


def _visualize_topic_model_3d(model, reduce_first=False, reducer="umap", port=8050):
    num_docs_per_topic = pd.Series(model.labels).value_counts().sort_index()

    # Extract top words for each topic with importance and format them vertically
    top_words_per_topic = {
        topic: "<br>".join(
            [f"{word} ({importance:.2f})" for word, importance in words[:5]]
        )
        for topic, words in model.output["topic_dict"].items()
    }

    if reducer == "umap":
        reducer = umap.UMAP(n_components=2)
    elif reducer == "tsne":
        reducer = TSNE(n_components=2, perplexity=30, learning_rate=200)
    elif reducer == "pca":
        reducer = PCA(n_components=2)
    else:
        raise ValueError("reducer must be in ['umap', 'tnse', 'pca']")

    if reduce_first:
        embeddings = reducer.fit_transform(model.embeddings)
        topic_data = []
        # Iterate over unique labels and compute mean embedding for each
        for label in np.unique(model.labels):
            # Find embeddings corresponding to the current label
            label_embeddings = embeddings[model.labels == label]
            # Compute mean embedding for the current label
            mean_embedding = np.mean(label_embeddings, axis=0)
            # Store the mean embedding in the dictionary
            topic_data.append(mean_embedding)
    else:
        if hasattr(model, "topic_centroids"):
            topic_data = model.topic_centroids
        else:
            topic_data = []
            # Iterate over unique labels and compute mean embedding for each
            for label in np.unique(model.labels):
                # Find embeddings corresponding to the current label
                label_embeddings = model.embeddings[model.labels == label]
                # Compute mean embedding for the current label
                mean_embedding = np.mean(label_embeddings, axis=0)
                # Store the mean embedding in the dictionary
                topic_data.append(mean_embedding)

    reducer = umap.UMAP(n_components=3)
    topic_embeddings_3d = reducer.fit_transform(topic_data)

    # Create DataFrame for plotting
    plot_df = pd.DataFrame(topic_embeddings_3d, columns=["x", "y", "z"])
    plot_df["topic"] = list(top_words_per_topic.keys())
    plot_df["num_docs"] = num_docs_per_topic
    plot_df["top_words"] = list(top_words_per_topic.values())

    app = dash.Dash(__name__)

    app.layout = html.Div(
        [
            html.H1("Topic Visualization"),
            dcc.Dropdown(
                id="side-dropdown",
                options=[
                    {"label": "Top Words", "value": "top_words"},
                    {"label": "Topic Distances", "value": "topic_distances"},
                ],
                value="top_words",
            ),
            dcc.Graph(id="main-plot"),
            dcc.Graph(id="side-plot"),
            html.Div(id="details-panel", children="Click a point for details"),
        ]
    )

    @app.callback(Output("main-plot", "figure"), [Input("side-dropdown", "value")])
    def update_main_plot(selected_option):
        fig = px.scatter_3d(
            plot_df,
            x="x",
            y="y",
            z="z",
            color="topic",
            size="num_docs",
            title="Main Topic Plot",
        )
        fig.update_layout(clickmode="event+select")
        return fig

    def get_top_words_for_topic(topic_number, model_output):
        return model_output["topic_dict"].get(topic_number, [])

    @app.callback(
        Output("side-plot", "figure"),
        [Input("main-plot", "clickData"), Input("side-dropdown", "value")],
    )
    def update_side_plot(clickData, side_option):
        if clickData:
            point_index = clickData["points"][0]["pointIndex"]
            selected_topic = point_index

            if side_option == "top_words":
                top_words_data = get_top_words_for_topic(selected_topic, model.output)
                words, scores = zip(*top_words_data)
                fig = px.bar(
                    x=words, y=scores, title=f"Top Words for Topic {selected_topic}"
                )
                return fig

            elif side_option == "topic_distances":
                distances_data = calculate_distances_to_other_topics(
                    point_index, plot_df, model.output
                )
                topics, distances, annotations = zip(*distances_data)
                fig = px.bar(
                    x=topics,
                    y=distances,
                    title=f"Distances from Topic {point_index} to Others",
                )
                # Add annotations
                fig.update_layout(
                    annotations=[
                        dict(
                            x=topic,
                            y=distance,
                            text=annotation,
                            showarrow=True,
                            arrowhead=1,
                            ax=0,
                            ay=-40,
                        )
                        for topic, distance, annotation in zip(
                            topics, distances, annotations
                        )
                    ]
                )

                fig.update_traces(
                    hovertemplate="<br>".join(
                        [
                            "Topic ID: %{x}",
                            "Distance: %{y:.2f}",
                            "<extra></extra>",  # Hides the trace name
                        ]
                    )
                )
                return fig

        return go.Figure()

    @app.callback(
        Output("details-panel", "children"), [Input("main-plot", "clickData")]
    )
    def display_details(clickData):
        if clickData:
            point_index = clickData["points"][0]["pointIndex"]
            selected_topic = point_index
            top_words_data = get_top_words_for_topic(selected_topic, model.output)
            detailed_info = "Topic {}: ".format(selected_topic) + ", ".join(
                [f"{word} ({score:.2f})" for word, score in top_words_data]
            )
            return detailed_info
        return "Click a point for details"

    app.run_server(debug=True, port=port)


def get_top_tfidf_words_per_document(corpus, n=10):
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()

    top_words_per_document = []
    for row in X:
        sorted_indices = np.argsort(row.toarray()).flatten()[::-1]
        top_n_indices = sorted_indices[:n]
        top_words = [(feature_names[i], row[0, i]) for i in top_n_indices]
        top_words_per_document.append(top_words)

    return top_words_per_document


def _visualize_topics_2d(model, reducer="umap", port=8050):
    
    embeddings = model.embeddings
    labels = model.labels
    top_words_per_document = get_top_tfidf_words_per_document(model.dataframe["text"])

    # Reduce embeddings to 2D for visualization
    if reducer == "umap":
        reducer = umap.UMAP(n_components=2)
    elif reducer == "tsne":
        reducer = TSNE(n_components=2, perplexity=30, learning_rate=200)
    elif reducer == "pca":
        reducer = PCA(n_components=2)
    else:
        raise ValueError("reducer must be in ['umap', 'tnse', 'pca']")
    reduced_embeddings = reducer.fit_transform(embeddings)

    # Prepare DataFrame for scatter plot
    df = pd.DataFrame(reduced_embeddings, columns=["x", "y"])
    df["label"] = labels
    df["top_words"] = [
        "<br>".join([f"{word} ({score:.2f})" for word, score in words])
        for words in top_words_per_document
    ]

    app = dash.Dash(__name__)

    # Dynamic slider for adjusting top words count
    app.layout = html.Div(
        [
            html.H1("Model Visualization Dashboard"),
            dcc.Slider(
                id="num-top-words-slider",
                min=1,
                max=25,
                value=5,
                marks={i: str(i) for i in range(1, 11)},
                step=1,
            ),
            dcc.Graph(id="scatter-plot"),
            html.Div(id="info-panel"),
        ]
    )

    @app.callback(
        Output("scatter-plot", "figure"), [Input("num-top-words-slider", "value")]
    )
    def update_plot(num_top_words):
        # Adjust top_words based on slider value
        df["top_words"] = [
            "<br>".join(
                [f"{word} ({score:.2f})" for word, score in words[:num_top_words]]
            )
            for words in top_words_per_document
        ]

        fig = px.scatter(
            df,
            x="x",
            y="y",
            color="label",
            hover_data={"top_words": True},
            labels={"color": "Cluster"},
            title="Document Clusters",
        )
        fig.update_traces(marker=dict(size=3))
        fig.update_layout(width=1000, height=800)
        return fig

    @app.callback(
        Output("info-panel", "children"), [Input("scatter-plot", "clickData")]
    )
    def display_info(clickData):
        # Display more information when a point is clicked
        if clickData:
            idx = clickData["points"][0]["pointIndex"]
            return html.P(f"Document ID: {idx}, More info here...")
        return "Click on a point to see more information."

    app.run_server(debug=True, port=port)


def _visualize_topics_3d(model, reducer="umap", port=8050):

    embeddings = model.embeddings
    labels = model.labels
    top_words_per_document = get_top_tfidf_words_per_document(model.dataframe["text"])

    if reducer == "umap":
        reducer = umap.UMAP(n_components=3)
    elif reducer == "tsne":
        reducer = TSNE(n_components=3, perplexity=30, learning_rate=200)
    elif reducer == "pca":
        reducer = PCA(n_components=3)
    else:
        raise ValueError("reducer must be in ['umap', 'tnse', 'pca']")
    reduced_embeddings = reducer.fit_transform(embeddings)

    # Prepare DataFrame for scatter plot
    df = pd.DataFrame(reduced_embeddings, columns=["x", "y", "z"])
    df["label"] = labels
    df["top_words"] = [
        "<br>".join([f"{word} ({score:.2f})" for word, score in words])
        for words in top_words_per_document
    ]

    # Initialize Dash app
    app = dash.Dash(__name__)

    # Create the figure using Plotly Express
    fig = px.scatter_3d(
        df,
        x="x",
        y="y",
        z="z",
        color="label",
        hover_data={"top_words": True},
        labels={"color": "Cluster"},
        title="Document Clusters",
    )

    # Manually adjust the marker size to make it smaller
    fig.update_traces(marker=dict(size=2))  # Decrease the marker size

    # Increase the overall plot size
    fig.update_layout(width=1000, height=800)  # Adjust width and height as needed

    app.layout = html.Div(
        [
            html.H1("Model Visualization Dashboard"),
            dcc.Graph(id="scatter-plot", figure=fig),
        ]
    )

    app.run_server(debug=True, port=port)
