import gradio as gr
import helpers as helpers
import pandas as pd
import numpy as np
import altair as alt
alt.data_transformers.disable_max_rows()

def main():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        with gr.Row():
            gr.Markdown("# Fractional Derivative Approximation Visualization\n \
                This demo visualizes the approximation of fractional \
                derivatives using many of the different approximation \
                methods.Such as Grünwald-Letnikov (GL) method.\n\n \
                Start by selecting a function to approximate the \
                derivative of")
        with gr.Row():
            with gr.Column(): # Input components

                func_dd = gr.Dropdown(label="Function", 
                                      info="Select a function to approximate the derivative of.",
                                      choices=list(helpers.FUNCTIONS.keys()), 
                                      value="x^2")
                deriv_order_num = gr.Number(label="Orders of Derivative",
                                            info="The order(s) of derivative to approximate.",
                                            value=1, 
                                            precision=0, 
                                            step=1, 
                                            minimum=0)
                deriv_frac_num = gr.Number(label="Number of Fractional Derivatives",
                                           info="The number of fractional derivatives to approximate between each integer order derivatives.",
                                           value = 3,
                                           precision=0,
                                           step=1,
                                           minimum=0)
            with gr.Column():
                num_samp_sl = gr.Slider(label="Number of Samples",
                                        info="The number of samples to generate.",
                                        value=300, 
                                        minimum =10, 
                                        maximum=2500, 
                                        precision=0)
                samp_step_sl = gr.Slider(label="Sample Step",
                                         info="The step size between each sample.",
                                         value=0.01, 
                                         minimum=0.00001,
                                         maximum=0.1)
                plot_width_sl = gr.Slider(label="Plot width",
                                          info="The width of the plot.",
                                          value=600, 
                                          minimum =250, 
                                          maximum=1920, 
                                          precision=0)
                plot_height_sl = gr.Slider(label="Plot width",
                            info="The width of the plot.",
                            value=400, 
                            minimum =250, 
                            maximum=1920, 
                            precision=0)
                gen_btn = gr.Button(label="Generate")
        with gr.Row(): # Output components
            plot = gr.LinePlot()

        gen_btn.click(generate_plot, 
                    inputs=[func_dd, deriv_order_num, deriv_frac_num, plot_width_sl, plot_height_sl, num_samp_sl, samp_step_sl],
                    outputs=plot)

    demo.queue().launch()

def generate_plot(function_sel, derive_order, num_deriv_frac, plot_width, plot_height, num_samples, sample_step, progress=gr.Progress(track_tqdm=True)):
    x, y = helpers.generate_function_data(helpers.FUNCTIONS[function_sel], num_samples, sample_step)
    df = pd.DataFrame({"X": x, "Y": y, "Order": function_sel, "Type": "Derivative"})
    
    # Generate derivative data
    for i in progress.tqdm(range(derive_order), desc="Generating integer order derivatives"):
        # First handle interger order derivatives
        # Error checking in case the user selects a derivative order that 
        # is not valid for the selected function
        try:
            y_d = helpers.get_deriv_approx(x, helpers.FUNCTIONS[function_sel], i+1)

            # If we got this far, integer order derivatives are valid, so now
            # do fractional order derivatives between
            frac_deriv_orders = np.linspace(start = i, 
                                            stop  = i+1,
                                            num   = num_deriv_frac+2)[1:-1]
            for j in progress.tqdm(frac_deriv_orders, desc="Generating fractional order derivatives"):
                y_d_frac = helpers.get_frac_deriv_approx_GL(x, helpers.FUNCTIONS[function_sel], j, sample_step)
                df_y = pd.DataFrame({"X": x, "Y": y_d_frac, "Order": f"{function_sel} Order ({j})", "Type": "Derivative (Fractional)"})
                df = pd.concat([df, df_y]) # Add to DataFrame

        except ValueError:
            gr.Warning(f"Derivative order {i+1} invalid for selected function.")
            break
        
        df_y = pd.DataFrame({"X": x, "Y": y_d, "Order": f"{function_sel} Order ({i+1})", "Type": "Derivative"})
        df = pd.concat([df, df_y]) # Add to DataFrame
    
    return gr.LinePlot.update(df, 
                              x="X", 
                              y="Y", 
                              color="Order", 
                              stroke_dash="Type", 
                              tooltip=["X", "Y", "Order"], 
                              title="Function Values", 
                              width=plot_width,
                              height=plot_height)

main()