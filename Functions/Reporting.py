import pandas as pd
import pandas.io.formats.style
import datetime as dt
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML

def write_to_html_file(df, title='', filename='out.html'):
    '''
    Write an entire dataframe to an HTML file with nice formatting.
    '''

    result = '''
<html>
<head>
<style>

    h2 {
        text-align: center;
        font-family: Helvetica, Arial, sans-serif;
    }
    table { 
        margin-left: auto;
        margin-right: auto;
    }
    table, th, td {
        border: 1px solid black;
        border-collapse: collapse;
    }
    th, td {
        padding: 5px;
        text-align: center;
        font-family: Helvetica, Arial, sans-serif;
        font-size: 90%;
    }
    table tbody tr:hover {
        background-color: #dddddd;
    }
    .wide {
        width: 90%; 
    }

</style>
</head>
<body>
    '''
    result += '<h2> %s </h2>\n' % title
    if type(df) == pd.io.formats.style.Styler:
        result += df.render()
    else:
        result += df.to_html(classes='wide', escape=False)
    result += '''
</body>
</html>
'''
    with open(filename, 'w') as f:
        f.write(result)


def PrintDescriptives(percentiles, save_path, save_name, df, cols, print_title, first=True,
                      round_int=2, start_time=dt.datetime.now()):
    descript = pd.DataFrame(round(df[cols].describe(percentiles=percentiles), round_int))
    if first == True:
        print(str(start_time) + "\n" + print_title +": ",
              file=open(str(save_path) + save_name, "w"))
    else:
        print("\n" + print_title + ": ",
              file=open(str(save_path) + save_name, "a"))

    print("Shape: cols= " + str(df.shape[1]) + ", rows= " + str(df.shape[0]),
          file=open(str(save_path) + save_name, "a"))
    print(descript,
          file=open(str(save_path) + save_name, "a"))
    return descript


def CreateReport(dict, template, pdf_save_name):
    env = Environment(loader=FileSystemLoader('.'))
    template = env.get_template(template)
    template_vars = dict
    # Render file and create the PDF
    html_out = template.render(template_vars)
    HTML(string=html_out).write_pdf()

    return

























