from pathlib import Path
from shiny import module, reactive, render, ui
from shinywidgets import output_widget, render_widget

@module.ui
def model3_ui():
    pass


@module.server
def model3_server(input, output, session, data_r, series_names_r):
    pass