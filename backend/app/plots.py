import plotly.graph_objects as go

def create_performance_plot(results, task, best_model_name):
    if not best_model_name or best_model_name == "No successful models":
        return None
    model_names = [n for n, r in results.items() if "error" not in r and "cv_" not in r]
    if not model_names:
        return None
    
    if task == "classification":
        fig = go.Figure(
            data=[
                go.Bar(name="Accuracy", x=model_names, y=[results[n]["accuracy"] for n in model_names]),
                go.Bar(name="F1 Macro", x=model_names, y=[results[n]["f1_macro"] for n in model_names]),
            ]
        )
        fig.update_layout(barmode="group", title="Classification Performance", yaxis_range=[0, 1])
    else:  # regression
        fig = go.Figure(
            data=[
                go.Bar(name="R²", x=model_names, y=[results[n]["r2"] for n in model_names]),
                go.Bar(name="RMSE", x=model_names, y=[results[n]["rmse"] for n in model_names], yaxis="y2"),
            ]
        )
        fig.update_layout(yaxis=dict(title="R²"), yaxis2=dict(title="RMSE", overlaying="y", side="right"))
    return fig.to_json()

def create_confusion_matrix(y_test, y_pred, model_name):
    if y_test is None or y_pred is None or len(np.unique(y_pred)) <= 1:
        return None
    cm = confusion_matrix(y_test, y_pred)
    fig = go.Figure(go.Heatmap(z=cm, colorscale="Blues"))
    fig.update_layout(title=f"Confusion Matrix - {model_name}")
    return fig.to_json()