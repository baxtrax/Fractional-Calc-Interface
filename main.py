import gradio as gr
import helpers as helpers
import pandas as pd

def generate_plot(function_sel, derive_order, num_samples, sample_step):
    x, y = helpers.generate_function_data(helpers.FUNCTIONS[function_sel], num_samples, sample_step)
    df = pd.DataFrame({"x": x, "y": y, "type": function_sel})
    # Ca
    for i in range(derive_order):
        try:
            y_d = helpers.get_deriv_approx(x, helpers.FUNCTIONS[function_sel], i+1)
        except ValueError:
            gr.Warning(f"Derivative order {i+1} invalid for selected function.")
            break
        df_y = pd.DataFrame({"x": x, "y": y_d, "type": f"{function_sel} Order ({i+1})"})
        df = pd.concat([df, df_y])

    return gr.LinePlot.update(df, x="x", y="y", color="type", height=300, width=500, tooltip=["x", "y"], title="Function Values",)

def main():
    with gr.Blocks() as demo:
        with gr.Column():
            func_dd = gr.Dropdown(label="Function", choices=list(helpers.FUNCTIONS.keys()), value="x^2")
            deriv_order_num = gr.Number(label="Orders of Derivative", value=1, precision=0, step=1, minimum=0)
            num_samp_num = gr.Number(label="Number of Samples", value=300)
            samp_step_sl = gr.Slider(label="Sample Step", value=0.01, minimum=0.01, maximum=0.1)
            gen_btn = gr.Button(label="Generate")
        with gr.Column():
            plot = gr.LinePlot()

        gen_btn.click(generate_plot, 
                    inputs=[func_dd, deriv_order_num, num_samp_num, samp_step_sl],
                    outputs=plot)

    demo.queue().launch()

main()