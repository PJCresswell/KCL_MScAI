import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import the reportlab document templates and style sheets
# The approach to build up the page content using an empty list and add content as flowables comes from the reportlab userguide
# https://docs.reportlab.com/reportlab/userguide/ch5_platypus/
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, NextPageTemplate, BaseDocTemplate, Frame, PageTemplate, Table, TableStyle
from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
sample_style_sheet = getSampleStyleSheet()

# Load in the environment variable for where to store the reports
from dotenv import load_dotenv
import os
load_dotenv()
report_dir = os.getenv('REPORT_DIR')

# Set the table style used within all reports
table_style = TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.blue),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 8),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 4),
    ('BACKGROUND', (0, 1), (-1, -1), colors.antiquewhite),
    ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
    ('ALIGN', (0, 1), (-1, -1), 'LEFT'),
    ('FONTNAME', (0, 1), (-1, -1), 'Times-Roman'),
    ('FONTSIZE', (0, 1), (-1, -1), 8),
    ('BOTTOMPADDING', (0, 1), (-1, -1), 4)
])

def process_news_article(article, ticker, content_flow):
    '''
    A function to process an individual news article to pull out the headline, sentiment and relevance score
    :param article: the article object
    :param ticker: the stock ticker of interest
    :param content_flow: the list variable where we are building up the report content
    :return: none
    '''
    # Each article, will be linked to a number of stock tickers. We look at each one in turn
    tickers = article.get('ticker_sentiment')
    for j in range(0, len(tickers)):
        company = tickers[j].get('ticker')
        # Check to see if this article refers to the stock we want
        if company == ticker:
            relevance = tickers[j].get('relevance_score')
            relevance_int = float(relevance)
            # Check to see if the relevance is above a set threshold. Gets rid of the noise
            if relevance_int >= 0.1:
                title = article.get('title')
                url = article.get('url')
                sentiment = tickers[j].get('ticker_sentiment_label')
                # Add to the report content variable
                paragraph_txt = Paragraph("Headline : " + title, sample_style_sheet['Heading3'])
                content_flow.append(paragraph_txt)
                paragraph_txt = Paragraph("Sentiment : " + str(sentiment) + ' : Relevance measure ' + str(round(relevance_int, 2)), sample_style_sheet['Bullet'])
                content_flow.append(paragraph_txt)
                paragraph_txt = Paragraph(url, sample_style_sheet['Italic'])
                content_flow.append(paragraph_txt)

def sentiment_report(ticker, news_date):
    '''
    A function to generate the sentiment report
    :param ticker: the stock ticker of interest
    :param news_date: the report execution date
    :return: none
    '''
    import requests

    # The list variable containing the report content. Built up as we go along
    content_flow = []
    # The API needs a range start and end date. Set both of these to be the explain date
    range_start_str = news_date + 'T0130'
    range_end_str = news_date + 'T0130'
    paragraph_txt = Paragraph("Sentiment Report for " + ticker + ' on ' + news_date, sample_style_sheet['Heading1'])
    content_flow.append(paragraph_txt)
    # Source the sentiment data on the explain date for this stock ticker
    url = ('https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=' + ticker + '&time_from=' + range_start_str + '&time_to=' + range_end_str + '&limit=1000&apikey=I2UR68ST032EG0J5')
    r = requests.get(url).json()
    # Pull the results into a dataframe
    df = pd.DataFrame.from_dict(r)
    # Check to see if anything has been returned
    if len(df) > 0:
        # Filter to just the date we want
        df['time'] = df['feed'].apply(lambda i: i.get('time_published'))
        df['date'] = df['time'].str[0:8]
        new_df = df[df['date'] == news_date]
        # Pull out the details for each news article returned
        # Vectorised call on each row in the dataframe calling the process_news_article function
        df['sentiment'] = new_df['feed'].apply(process_news_article, args=(ticker, content_flow))
    # Save down the report created
    pdf = SimpleDocTemplate(report_dir + "Sentiment_Report_" + ticker + "_" + news_date + ".pdf")
    pdf.build(content_flow)

def daily_report(date, ticker, t, tminus1, context, comp_date, matchrate, window1, window2, window3, forecast):
    '''
    A function to generate the tables showing the input data for a specific date.
    Used by the Daily Report to show the input data which went into the prediction.
    Used by the Similarity Report to show the input data on the date deemed to have the closest match.
    :param date: the report execution date
    :param ticker: the stock ticker of interest
    :param t: the input data for the date on which the report is being run
    :param tminus1: the input data for the day before the report execution date
    :param context: which report is required - Daily or Similarity
    :param comp_date: for the similarity report - the comparison date
    :param matchrate: for the similarity report - the match rate
    :param window1: for the similarity report - the EOD price trend leading up to the model date
    :param window2: for the similarity report - the EOD price trend leading up to the identified similar date
    :param window3: for the similarity report - the EOD price trend following the identified similar date
    :param forecast: the model prediction
    :return: none
    '''
    # Helper functions used when creating the report for features which put values into one of a number of bins
    # Creates a blank mask and then inserts the feature into the appropriate slot
    def allocate_from_3_bins(t, feature, feature_name):
        mask = [' ', ' ', ' ']
        bin = t[feature][0]
        mask[bin - 1] = feature_name
        return mask
    def allocate_from_4_bins(t, feature, feature_name):
        mask = [' ', ' ', ' ', ' ']
        bin = t[feature][0]
        mask[bin - 1] = feature_name
        return mask
    # The list variable containing the report content. Built up as we go along
    content_flow = []
    if context == 'SIMILAR':
        # Create the report header if we're running a similarity report
        pdf = SimpleDocTemplate(report_dir + "Similarity_Report_" + ticker + "_" + comp_date + ".pdf")
        paragraph_txt = Paragraph("Similarity Report for " + ticker + " on " + comp_date, sample_style_sheet['Heading1'])
        content_flow.append(paragraph_txt)
        # The most important thing here is to provide context to the user around the comparison
        # We start by showing the price trend for the seven days leading up to the prediction date
        # Saves down as an image which is then added to the report flow
        paragraph_txt = Paragraph("EOD return trend leading up to model date", sample_style_sheet['Heading3'])
        content_flow.append(paragraph_txt)
        plt.bar(window1.index, window1)
        plt.savefig('img1.png', bbox_inches="tight")
        plt.clf()
        image1 = Image('img1.png', width=300, height=150)
        content_flow.append(image1)
        # We then output the recommendation based on the model prediction
        if forecast == 1:
            paragraph_txt = Paragraph("Recommendation : Go long", sample_style_sheet['Heading3'])
        else:
            paragraph_txt = Paragraph("Recommendation : Go short", sample_style_sheet['Heading3'])
        content_flow.append(paragraph_txt)
        # We then output the most similar date found along with the match rate
        # If the match rate is low then an example like today was not found in the training set
        paragraph_txt = Paragraph("Most similar date in training set : " + date, sample_style_sheet['Heading3'])
        content_flow.append(paragraph_txt)
        paragraph_txt = Paragraph("Match % of " + str(round(matchrate[0], 2)), sample_style_sheet['Heading3'])
        content_flow.append(paragraph_txt)
        # For comparison we then show the price trend for the seven days leading up to the most similar date
        # Saves down as an image which is then added to the report flow
        paragraph_txt = Paragraph("EOD return trend leading up to " + date, sample_style_sheet['Heading3'])
        content_flow.append(paragraph_txt)
        plt.bar(window2.index, window2[ticker + '_Return'])
        plt.savefig('img2.png', bbox_inches="tight")
        plt.clf()
        image2 = Image('img2.png', width=300, height=150)
        content_flow.append(image2)
        # And finally we show the price trend for the seven days following the most similar date
        # Saves down as an image which is then added to the report flow
        paragraph_txt = Paragraph("EOD return trend following " + date, sample_style_sheet['Heading3'])
        content_flow.append(paragraph_txt)
        plt.bar(window3.index, window3[ticker + '_Return'])
        plt.savefig('img3.png', bbox_inches="tight")
        plt.clf()
        image3 = Image('img3.png', width=300, height=150)
        content_flow.append(image3)
        paragraph_txt = Paragraph('Input data on ' + date, sample_style_sheet['Heading2'])
        content_flow.append(paragraph_txt)
    else:
        # Create the report header if we're running a Daily Report
        pdf = SimpleDocTemplate(report_dir + "Daily_Report_" + ticker + "_" + date + ".pdf")
        paragraph_txt = Paragraph("Daily report for " + ticker + ' on ' + date, sample_style_sheet['Heading1'])
        content_flow.append(paragraph_txt)
        # Output the recommendation based on the model prediction
        if forecast == 1:
            paragraph_txt = Paragraph("Recommendation : Go long", sample_style_sheet['Heading2'])
        else:
            paragraph_txt = Paragraph("Recommendation : Go short", sample_style_sheet['Heading2'])
        content_flow.append(paragraph_txt)
    # Show the input data used by the model in a user-friendly format
    # Will either be the prediction date or the similarity date based on the context from which the function was called
    paragraph_txt = Paragraph("COB Market Metrics", sample_style_sheet['Heading3'])
    content_flow.append(paragraph_txt)
    paragraph_txt = Paragraph("Direction/Trend is day over day", sample_style_sheet['Heading5'])
    content_flow.append(paragraph_txt)
    # The first table displays the core market metrics along with their direction or trend
    table1 = []
    table1.append(['Feature', 'Value', 'Direction/Trend'])
    table1.append(['EOD Close', round(t[ticker][0], 2), np.sign(t[ticker][0] - tminus1[ticker][0])])
    table1.append(['Log Return', round(t[ticker + '_Return'][0], 2), np.sign(t[ticker + '_Return'][0] - tminus1[ticker + '_Return'][0])])
    table1.append(['S&P 500', round(t['^GSPC'][0], 2), np.sign(t['^GSPC'][0] - tminus1['^GSPC'][0])])
    table1.append(['VIX', round(t['^VIX'][0], 2), np.sign(t['^VIX'][0] - tminus1['^VIX'][0])])
    table_final = Table(table1)
    table_final.setStyle(table_style)
    content_flow.append(table_final)
    # The second table displays the moving average and lagged return data which has been bucketed related to the mean and moving average
    # I decided not to include the actual values as makes too busy - losing the overall messaging
    paragraph_txt = Paragraph("COB calculated measures, bucketed wrt the overall mean and SD", sample_style_sheet['Heading3'])
    content_flow.append(paragraph_txt)
    table2 = []
    table2.append(['< mean - SD', '< mean', '> mean', '> mean + SD'])
    table2.append(allocate_from_4_bins(t, ticker+'_SMA_sht_bin', 'SMA short'))
    table2.append(allocate_from_4_bins(t, ticker+'_SMA_lng_bin', 'SMA long'))
    table2.append(allocate_from_4_bins(t, 'SNP_SMA_sht_bin', 'SnP SMA short'))
    table2.append(allocate_from_4_bins(t, 'SNP_SMA_lng_bin', 'SnP SMA long'))
    table2.append(allocate_from_4_bins(t, 'VIX_SMA_sht_bin', 'VIX SMA short'))
    table2.append(allocate_from_4_bins(t, 'VIX_SMA_lng_bin', 'VIX SMA long'))
    table2.append(allocate_from_4_bins(t, ticker + '_lag_1_bin', '1d Lag Returns'))
    table2.append(allocate_from_4_bins(t, ticker + '_lag_2_bin', '2d Lag Returns'))
    table2.append(allocate_from_4_bins(t, ticker + '_lag_3_bin', '3d Lag Returns'))
    table2.append(allocate_from_4_bins(t, ticker + '_lag_4_bin', '4d Lag Returns'))
    table2.append(allocate_from_4_bins(t, ticker + '_lag_5_bin', '5d Lag Returns'))
    table_final = Table(table2)
    table_final.setStyle(table_style)
    content_flow.append(table_final)
    # The third table displays the technical metrics which are included in the model relating to direction of movement or trend
    # The only exception are the Aroon up and down values. The difference is used by the model but there is value to the user in showing both
    paragraph_txt = Paragraph("Technical metrics, included in the model wrt trend", sample_style_sheet['Heading3'])
    content_flow.append(paragraph_txt)
    paragraph_txt = Paragraph("Chaikin A/D line + On balance volume : Trend over the last 3 days", sample_style_sheet['Heading5'])
    content_flow.append(paragraph_txt)
    paragraph_txt = Paragraph("Parabolic SAR flag : Difference between Parabolic SAR and the EOD close. 1 if positive, -1 if negative", sample_style_sheet['Heading5'])
    content_flow.append(paragraph_txt)
    paragraph_txt = Paragraph("Aroon Up/Down values : Difference day on day", sample_style_sheet['Heading5'])
    content_flow.append(paragraph_txt)
    table3 = []
    table3.append(['Feature', 'Value', 'Direction/Trend/Difference'])
    table3.append(['Chaikin A/D line (AD)', round(t[ticker+'_AD'][0], 2), t[ticker+'_AD_trend'][0]])
    table3.append(['On balance volume (OBV)', round(t[ticker+'_OBV'][0], 2), t[ticker+'_OBV_trend'][0]])
    table3.append(['Parabolic SAR (SAR)', round(t[ticker+'_SAR'][0], 2), t[ticker + '_SAR_Flag'][0]])
    table3.append(['Aroon (AROON) UP values', round(t[ticker+'_AR_UP'][0], 2), t[ticker+'_AR_UP'][0] - tminus1[ticker+'_AR_UP'][0]])
    table3.append(['Aroon (AROON) DOWN values', round(t[ticker+'_AR_DN'][0], 2), t[ticker+'_AR_DN'][0] - tminus1[ticker+'_AR_DN'][0]])
    table_final = Table(table3)
    table_final.setStyle(table_style)
    content_flow.append(table_final)
    # The last four tables display the technical metrics which are included in the model relating to specific bucketed values
    # Before each table, text is added to remind the user how to read the results
    paragraph_txt = Paragraph("Technical metrics, included in the model wrt bucketed value", sample_style_sheet['Heading3'])
    content_flow.append(paragraph_txt)
    paragraph_txt = Paragraph("Ultimate Oscillator + Relative Strength Index : Values over 70 indicate overbought conditions, values under 30 indicate oversold conditions", sample_style_sheet['Heading5'])
    content_flow.append(paragraph_txt)
    table4 = []
    table4.append(['0 - 40', '40 - 60', '60-100'])
    table4.append(allocate_from_3_bins(t, ticker+'_ULT_BIN', 'Ult Osc'))
    table4.append(allocate_from_3_bins(t, ticker+'_RSI_BIN', 'RSI'))
    table_final = Table(table4)
    table_final.setStyle(table_style)
    content_flow.append(table_final)
    paragraph_txt = Paragraph("Directional Movement Index + Average Directional Movement Index : Typically 0-60 with a high number being a strong trend. Values over 70 indicate overbought conditions. Values under 30 indicate oversold conditions", sample_style_sheet['Heading5'])
    content_flow.append(paragraph_txt)
    table5 = []
    table5.append(['0 - 20', '20 - 40', '40 - 60', '60-100'])
    table5.append(allocate_from_4_bins(t, ticker + '_DX_BIN', 'DX'))
    table5.append(allocate_from_4_bins(t, ticker+'_ADX_BIN', 'ADX'))
    table_final = Table(table5)
    table_final.setStyle(table_style)
    content_flow.append(table_final)
    paragraph_txt = Paragraph("Williams %R values : Below 20 indicates overbought condition. Values over 80 indicate an oversold condition", sample_style_sheet['Heading5'])
    content_flow.append(paragraph_txt)
    table6 = []
    table6.append(['-100 to -80', '-80 to -20', '-20 to 0', '0'])
    table6.append(allocate_from_4_bins(t, ticker+'_WIL_BIN', 'WILLIAMS %R'))
    table_final = Table(table6)
    table_final.setStyle(table_style)
    content_flow.append(table_final)
    paragraph_txt = Paragraph("Money Flow Index : Values above 80 indicate market tops. Values below 20 indicate market bottoms", sample_style_sheet['Heading5'])
    content_flow.append(paragraph_txt)
    table7 = []
    table7.append(['0 to 30', '30 to 70', '70 to 100', '100'])
    table7.append(allocate_from_4_bins(t, ticker+'_MFI_BIN', 'MFI'))
    table_final = Table(table7)
    table_final.setStyle(table_style)
    content_flow.append(table_final)
    # Save down the report
    pdf.build(content_flow)
    # Tidy up by removing the images
    if context == 'SIMILAR':
        os.remove("img1.png")
        os.remove("img2.png")
        os.remove("img3.png")

def anchors_report(date, ticker, rule, precision, coverage, forecast):
    '''
    A function to generate the report of the results of the Anchors Decision Rule explain
    :param date: the report execution date
    :param ticker: the ticker of interest
    :param rule: the anchor rule
    :param precision: the anchor rule precision
    :param coverage: the anchor rule coverage
    :param forecast: the model movement prediction
    :return: none
    '''
    # The list variable containing the report content. Built up as we go along
    content_flow = []
    paragraph_txt = Paragraph("Anchors Report for " + ticker + ' on ' + date, sample_style_sheet['Heading1'])
    content_flow.append(paragraph_txt)
    # Output the recommendation based on the model prediction
    if forecast == 1:
        paragraph_txt = Paragraph("Recommendation : Go long", sample_style_sheet['Heading2'])
    else:
        paragraph_txt = Paragraph("Recommendation : Go short", sample_style_sheet['Heading2'])
    content_flow.append(paragraph_txt)
    # Output the decision rule generated
    paragraph_txt = Paragraph("Decision rule anchoring the forecast", sample_style_sheet['Heading2'])
    content_flow.append(paragraph_txt)
    paragraph_txt = Paragraph(rule, sample_style_sheet['Heading3'])
    content_flow.append(paragraph_txt)
    # Output the confidence and precision of the decision rule
    paragraph_txt = Paragraph("Confidence :", sample_style_sheet['Heading2'])
    content_flow.append(paragraph_txt)
    paragraph_txt = Paragraph("Precision " + str(round(precision, 2)) + ", Coverage " + str(round(coverage, 2)), sample_style_sheet['Heading3'])
    content_flow.append(paragraph_txt)
    # Save down the report
    pdf = SimpleDocTemplate(report_dir + "Anchors_Report_" + ticker + "_" + date + ".pdf")
    pdf.build(content_flow)

def DICE_report(counterfactuals, original, forecast, date, ticker, columns, filter):
    '''
    A function to generate the report containing the counterfactual results
    :param counterfactuals: the counterfactual examples generated
    :param original: the original input data which went into the model prediction
    :param forecast: the model movement prediction
    :param date: the report execution date
    :param ticker: the ticker of interest
    :param columns: the list of columns included within the counterfactual generation
    :param filter: context for the report - whether the counterfactuals include all features or a user filtered set
    :return: none
    '''
    import math

    newdf = counterfactuals.cf_examples_list[0].final_cfs_df.values.tolist()
    # In the report, I only want to show the changed values - if not changed, show as a "-" for clarity
    # I therefore need to go through each element in each counterfactual example in turn to see if different
    # The function visualize_as_dataframe() within the DICE package can do this but only from within the IPython / Jupyter environment
    # The code below comes from display_df() which is called by visualize_as_dataframe() and does just what I need
    # https: // interpret.ml / DiCE / dice_ml.html  # dice_ml.diverse_counterfactuals.CounterfactualExamples.visualize_as_dataframe
    for ix in range(counterfactuals.cf_examples_list[0].final_cfs_df.shape[0]):
        for jx in range(len(original[0])):
            if not isinstance(newdf[ix][jx], str):
                if math.isclose(newdf[ix][jx], original[0][jx], abs_tol=0.01):
                    newdf[ix][jx] = '-'
                else:
                    newdf[ix][jx] = str(newdf[ix][jx])
            else:
                if newdf[ix][jx] == original[0][jx]:
                    newdf[ix][jx] = '-'
                else:
                    newdf[ix][jx] = str(newdf[ix][jx])
    # Add the actual model prediction to the end of the dataframe
    results = pd.DataFrame(original + newdf, columns=columns + ['Forecast'])
    # Transpose so the features are rows and each counterfactual example is a column
    results_tran = results.transpose()
    # Add a column containing a short description of each feature and what it's measuring re bucketing, trends etc
    results_tran['Feature Attribute'] = ['1d lagged returns (bin num)\n< mean - SD; < mean; > mean; > mean + SD',
                                         '2d lagged returns (bin num)\n< mean - SD; < mean; > mean; > mean + SD',
                                         '3d lagged returns (bin num)\n< mean - SD; < mean; > mean; > mean + SD',
                                         '4d lagged returns (bin num)\n< mean - SD; < mean; > mean; > mean + SD',
                                         '5d lagged returns (bin num)\n< mean - SD; < mean; > mean; > mean + SD',
                                         'Chaikin AD trend\nUP(+1), DN(-1)',
                                         'Ultimate Oscillator (bin num)\n0 - 40; 40 - 60; 60-100',
                                         'Relative strength index (bin num)\n0 - 40; 40 - 60; 60-100',
                                         'Williams Value (bin num)\n-100 to -80; -80 to -20; -20 to 0; 0',
                                         'Avg directional movement (bin num)\n0 - 20; 20 - 40; 40 - 60; 60-100',
                                         'Aroon value trend\nUP(+1), DN(-1)',
                                         'Money flow index (bin num)\n0 to 30; 30 to 70; 70 to 100; 100',
                                         'Directional movement (bin num)\n0 - 20; 20 - 40; 40 - 60; 60-100',
                                         'On balance volume trend\nUP(+1), DN(-1)',
                                         'Parabolic SAR flag\n+ or - diff to EOD Close',
                                         'Short stock SMA trend (bin num)\n< mean - SD; < mean; > mean; > mean + SD',
                                         'Long stock SMA trend (bin num)\n< mean - SD; < mean; > mean; > mean + SD',
                                         'Short SnP SMA trend (bin num)\n< mean - SD; < mean; > mean; > mean + SD',
                                         'Long SnP SMA trend (bin num)\n< mean - SD; < mean; > mean; > mean + SD',
                                         'Short VIX SMA trend (bin num)\n< mean - SD; < mean; > mean; > mean + SD',
                                         'Long VIX SMA trend (bin num)\n< mean - SD; < mean; > mean; > mean + SD',
                                         'Model forecast\nLong(+1), Short(-1)']
    # Make this column containing the feature descriptions the first one in the table
    col = results_tran.columns.tolist()
    col.insert(0, col.pop())
    results_tran = results_tran[col]
    # The list variable containing the report content. Built up as we go along
    content_flow = []
    paragraph_1 = Paragraph("Counterfactual examples", sample_style_sheet['Heading1'])
    content_flow.append(paragraph_1)
    paragraph_1 = Paragraph(ticker + " on " + date, sample_style_sheet['Heading2'])
    content_flow.append(paragraph_1)
    # Remind the user of the prediction made and so what this counterfactual example is telling them
    paragraph_1 = Paragraph("to get the opposite prediction from model result of " + str(forecast), sample_style_sheet['Heading2'])
    content_flow.append(paragraph_1)
    # This function is called under two contexts : Add a row to the report explaining which one
    if filter == 'ALL':
        # Counterfactuals generated with no restrictions
        paragraph_1 = Paragraph("No user restrictions", sample_style_sheet['Heading3'])
        pdf = SimpleDocTemplate(report_dir + "Counterfactuals_All_" + ticker + "_" + date + ".pdf")
    else:
        # Counterfactuals generated under user restrictions
        paragraph_1 = Paragraph("Restricted to varying : " + filter, sample_style_sheet['Heading3'])
        pdf = SimpleDocTemplate(report_dir + "Counterfactuals_Filtered_" + ticker + "_" + date + ".pdf")
    content_flow.append(paragraph_1)
    # Finally, convert the dataframe containing the counterfactual examples to a table able to be added to the report
    table1_data = []
    table1_data.append(['Feature attributes', 'Input data', 'CF 1', 'CF 2', 'CF 3', 'CF 4', 'CF 5'])
    for i, row in results_tran.iterrows():
        table1_data.append(list(row))
    table1_data = table1_data[:-1]
    table_final = Table(table1_data)
    table_final.setStyle(table_style)
    content_flow.append(table_final)
    # Save down the report
    pdf.build(content_flow)

def SHAP_report(shap_values, ticker, explain_date):
    '''
    A function to generate both the individual example SHAP report, and the model level SHAP report
    :param shap_values: the explains object containing the generated SHAP values
    :param ticker: the ticker of interest
    :param explain_date: the report execution date
    :return: none
    '''
    import shap

    # During development, the SHAP reporting library functions very occasionally raised errors
    # I have therefore wrapped this function in an exception handling framework to ensure the overall batch run is never affected
    try:
        # We generate first the model level report
        # The list variable containing the report content. Built up as we go along
        content_flow = []
        paragraph_txt = Paragraph("SHAP feature attribution", sample_style_sheet['Heading1'])
        content_flow.append(paragraph_txt)
        paragraph_txt = Paragraph("Model level results for " + ticker + " on " + explain_date, sample_style_sheet['Heading1'])
        content_flow.append(paragraph_txt)
        # The report consists of three plots. Each is generated and saved down as an image before being loaded into the report content variable
        # Includes the values across all examples included within the shap_values results
        # The bar plot
        paragraph_txt = Paragraph("Bar plot", sample_style_sheet['Heading3'])
        content_flow.append(paragraph_txt)
        shap.plots.bar(shap_values, show=False)
        plt.savefig('img1.png', bbox_inches="tight")
        plt.clf()
        image1 = Image('img1.png', width=300, height=150)
        content_flow.append(image1)
        # The beeswarm plot
        paragraph_txt = Paragraph("Beeswarm plot", sample_style_sheet['Heading3'])
        content_flow.append(paragraph_txt)
        shap.plots.beeswarm(shap_values, show=False)
        plt.savefig('img2.png', bbox_inches="tight")
        plt.clf()
        image2 = Image('img2.png', width=300, height=150)
        content_flow.append(image2)
        # The summary plot
        paragraph_txt = Paragraph("Summary plot", sample_style_sheet['Heading3'])
        content_flow.append(paragraph_txt)
        shap.summary_plot(shap_values, plot_type='violin', show=False)
        plt.savefig('img3.png', bbox_inches="tight")
        plt.clf()
        image3 = Image('img3.png', width=300, height=200)
        content_flow.append(image3)
        # Build the report
        pdf = SimpleDocTemplate(report_dir + "Model_SHAP_Values_" + ticker + "_" + explain_date + ".pdf")
        pdf.build(content_flow)
        # Tidy up by removing the images
        os.remove("img1.png")
        os.remove("img2.png")
        os.remove("img3.png")

        # Now produce the SHAP report for the specific explain date
        content_flow = []
        paragraph_txt = Paragraph("SHAP feature attribution", sample_style_sheet['Heading1'])
        content_flow.append(paragraph_txt)
        paragraph_txt = Paragraph("Drivers behind the forecast for " + ticker + " on " + explain_date, sample_style_sheet['Heading1'])
        content_flow.append(paragraph_txt)
        # Again the report consists of three plots. Each is generated and saved down as an image before being loaded into the report content variable
        # Includes values for just the last example included within the shap_values results - ie today, the explain date
        # The bar plot
        paragraph_txt = Paragraph("Bar plot", sample_style_sheet['Heading3'])
        content_flow.append(paragraph_txt)
        shap.plots.bar(shap_values[-1], show=False)
        plt.savefig('img4.png', bbox_inches="tight")
        plt.clf()
        image4 = Image('img4.png', width=300, height=150)
        content_flow.append(image4)
        # The waterfall plot
        paragraph_txt = Paragraph("Waterfall plot", sample_style_sheet['Heading3'])
        content_flow.append(paragraph_txt)
        shap.plots.waterfall(shap_values[-1], show=False)
        plt.savefig('img5.png', bbox_inches="tight")
        plt.clf()
        image5 = Image('img5.png', width=300, height=150)
        content_flow.append(image5)
        # The force plot
        paragraph_txt = Paragraph("Force plot", sample_style_sheet['Heading3'])
        content_flow.append(paragraph_txt)
        shap.plots.force(shap_values[-1], matplotlib=True, show=False)
        plt.savefig('img6.png', bbox_inches="tight")
        plt.clf()
        image6 = Image('img6.png', width=300, height=150)
        content_flow.append(image6)
        # Build the report
        pdf = SimpleDocTemplate(report_dir + "SHAP_Values_" + ticker + "_" + explain_date + ".pdf")
        pdf.build(content_flow)
        # Tidy up by removing the images
        os.remove("img4.png")
        os.remove("img5.png")
        os.remove("img6.png")
    except Exception as ex:
        print('Issue generating the SHAP reports for ' + ticker + ' : Code ' + str(ex))

def history_report(test, ticker, date):
    '''
    A function to generate the trade history report
    :param test: the testing dataset
    :param ticker: the ticker of interest
    :param date: the report execution date
    :return: none
    '''
    # The list variable containing the report content. Built up as we go along
    content_flow = []
    paragraph_txt = Paragraph('Trade history report for ' + ticker + ' as of ' + date, sample_style_sheet['Heading1'])
    content_flow.append(paragraph_txt)
    # First, creates a plot of the EOD price across the testing period
    paragraph_txt = Paragraph('Long term return trend', sample_style_sheet['Heading2'])
    content_flow.append(paragraph_txt)
    test[ticker + '_Return'].cumsum().apply(np.exp).plot(figsize=(10, 6))
    plt.savefig('img1.png', bbox_inches="tight")
    plt.clf()
    image1 = Image('img1.png', width=400, height=300)
    content_flow.append(image1)
    # Add a table containing the detailed information over the last 20 days
    # Includes the predicted model position
    # Includes a placeholder to store the actual position taken - will come from integration with a "real" trading system
    paragraph_txt = Paragraph('Detailed view over the last 20 days', sample_style_sheet['Heading2'])
    content_flow.append(paragraph_txt)
    paragraph_txt = Paragraph('EOD prices, daily returns and positions taken', sample_style_sheet['Heading3'])
    content_flow.append(paragraph_txt)
    test['date_field'] = test.index
    test['rnd_cob_price'] = round(test[ticker], 2)
    test['rnd_return'] = round(test[ticker + '_Return'], 2)
    table1_data = []
    table1_data.append(['Date', 'EOD Price', 'Daily Return', 'Model Position', 'Actual Position'])
    for i, row in test[['date_field', 'rnd_cob_price', 'rnd_return', ticker+'_model_position']].tail(20).iterrows():
        table1_data.append(list(row))
    table_final = Table(table1_data)
    table_final.setStyle(table_style)
    content_flow.append(table_final)
    pdf = SimpleDocTemplate(report_dir + "Trade_History_" + ticker + "_" + date + ".pdf")
    # Build the report
    pdf.build(content_flow)
    # Tidy up by removing the image
    os.remove("img1.png")

def portfolio_report(portfolio, date, position_results, cleaned_weights, mu, S, ef, portfolio_value, allocation, leftover):
    '''
    A function to generate the portfolio optimisation report
    :param portfolio: the list of stocks to include within the portfolio optimisation
    :param date: the report execution date
    :param position_results: the results of the model movement predictions for the date (t) and the day before (t-1)
    :param cleaned_weights: the cleaned weights for te calculated max sharpe value portfolio
    :param mu: the expected returns for the portfolio
    :param S: the sample covariance for the portfolio
    :param ef: the efficient frontier object
    :param portfolio_value: the funds available for allocation
    :param allocation: what that discrete allocation is
    :param leftover: funds left over
    :return: none
    '''

    from pypfopt import EfficientFrontier
    from pypfopt import plotting

    # We start by generating the header for the report
    # The list variable containing the report content. Built up as we go along
    content_flow = []
    paragraph_txt = Paragraph("Portfolio report as of " + date, sample_style_sheet['Heading1'])
    content_flow.append(paragraph_txt)
    # Lists the stocks included - is configurable by the user in the environment file
    paragraph_txt = Paragraph("Portfolio : " + str(portfolio), sample_style_sheet['Heading3'])
    content_flow.append(paragraph_txt)
    # For each stock, output the model movement predictions for the explain date (today) and the previous date (yesterday)
    paragraph_txt = Paragraph("Model movement predictions - today and yesterday", sample_style_sheet['Heading3'])
    content_flow.append(paragraph_txt)
    # For each stock, output the model movement predictions for the explain date (today) and the previous date (yesterday)
    table1_data = []
    table1_data.append(['Stock Ticker', 'Date', 'Movement prediction', 'Previous date', 'Previous Movement prediction'])
    for i, row in position_results.iterrows():
        table1_data.append(list(row))
    table_final = Table(table1_data)
    table_final.setStyle(table_style)
    content_flow.append(table_final)

    # Now add the results of the portfolio optimisation
    paragraph_txt = Paragraph("Portfolio weights", sample_style_sheet['Heading3'])
    content_flow.append(paragraph_txt)
    # Print our the portfolio weights
    paragraph_txt = Paragraph(str(cleaned_weights).strip('OrderedDict'), sample_style_sheet['Normal'])
    content_flow.append(paragraph_txt)
    # Also show the weights graphically
    plotting.plot_weights(cleaned_weights)
    plt.savefig('img1.png', bbox_inches="tight")
    plt.clf()
    image1 = Image('img1.png', width=250, height=250)
    content_flow.append(image1)

    # Print out the various portfolio portfolio statistics available
    paragraph_txt = Paragraph("Portfolio details", sample_style_sheet['Heading3'])
    content_flow.append(paragraph_txt)
    er, vol, sharpe = ef.portfolio_performance()
    paragraph_txt = Paragraph('Expected annual return (%): ' + str(round(er, 2)), sample_style_sheet['Normal'])
    content_flow.append(paragraph_txt)
    paragraph_txt = Paragraph('Annual volatility (%) ' + str(round(vol, 2)), sample_style_sheet['Normal'])
    content_flow.append(paragraph_txt)
    paragraph_txt = Paragraph('Sharpe Ratio: ' + str(round(sharpe, 2)), sample_style_sheet['Normal'])
    content_flow.append(paragraph_txt)

    # Output to the report the discrete allocation along with any remaining funds
    paragraph_txt = Paragraph("Remaining funds from portfolio of $"+ portfolio_value + " : ${:.2f}".format(leftover), sample_style_sheet['Normal'])
    content_flow.append(paragraph_txt)
    paragraph_txt = Paragraph("Discrete allocation : " + str(allocation), sample_style_sheet['Normal'])
    content_flow.append(paragraph_txt)

    # Finally, two plots provided by the package supporting the calculation
    fig, ax = plt.subplots()
    paragraph_txt = Paragraph("Portfolio efficient frontier", sample_style_sheet['Heading3'])
    content_flow.append(paragraph_txt)
    # Firstly, the efficient frontier
    # Shows the individual stock volatility and returns - along with the set of optimal portfolios which minimise the risk for a target return
    ef = EfficientFrontier(mu, S, weight_bounds=(-1, 1))
    ef1 = ef.deepcopy()
    plotting.plot_efficient_frontier(ef, ax=ax, ef_param='return', show_tickers=True)
    ef1.max_sharpe()
    ret_tangent, std_tangent, _ = ef1.portfolio_performance()
    ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Max Sharpe")
    plt.savefig('img2.png', bbox_inches="tight")
    plt.clf()
    image2 = Image('img2.png', width=250, height=250)
    content_flow.append(image2)

    # Next the covariance matrix across the stocks within the portfolio
    paragraph_txt = Paragraph("Portfolio covariance", sample_style_sheet['Heading3'])
    content_flow.append(paragraph_txt)
    plotting.plot_covariance(S)
    plt.savefig('img3.png', bbox_inches="tight")
    plt.clf()
    image3 = Image('img3.png', width=250, height=250)
    content_flow.append(image3)
    # Build the report
    pdf = SimpleDocTemplate(report_dir + 'Portfolio_Report_' + date + '.pdf')
    pdf.build(content_flow)
    # Tidy up afterwards by removing the images
    os.remove("img1.png")
    os.remove("img2.png")
    os.remove("img3.png")

def performance_report(model, train, test, columns, ticker, date):
    '''
    A function to generate the model performance report
    :param model: a model object
    :param train: the training dataset
    :param test: the testing dataset
    :param columns: the columns used to train the model
    :param ticker: the ticker of interest
    :param date: the report execution date
    :return: none
    '''
    # The list variable containing the report content. Built up as we go along
    content_flow = []
    paragraph_txt = Paragraph('Model performance report for ' + ticker + ' as of ' + date, sample_style_sheet['Heading1'])
    content_flow.append(paragraph_txt)

    # My approach uses vectorized backtesting to assess model performance (multiplying the log returns by the positions) as well as
    # comparing against simply holding the stock over the horizon. Adapted from the book "Python for Finance" (Yves Hilpisch, 2019) Section 15 : Trading Strategies

    # The model score on the training set
    paragraph_txt = Paragraph('Training set model score = ' + str(round(model.score(train[columns], train[ticker + '_Direction_bin']), 2)), sample_style_sheet['Heading2'])
    content_flow.append(paragraph_txt)
    # The model return on the training set along with that for just holding the stock
    # The plot across the window is generated and saved down before being loaded into the report flow object
    train[[ticker + '_Return', ticker + '_model_results']].cumsum().apply(np.exp).plot(figsize=(10, 6))
    plt.savefig('img1.png', bbox_inches="tight")
    plt.clf()
    image1 = Image('img1.png', width=200, height=200)
    content_flow.append(image1)

    # The model score on the testing set
    paragraph_txt = Paragraph('Testing set model score = ' + str(round(model.score(test[columns], test[ticker + '_Direction_bin']), 2)), sample_style_sheet['Heading2'])
    content_flow.append(paragraph_txt)
    # The model return on the testing set along with that for just holding the stock
    test[[ticker + '_Return', ticker + '_model_results']].cumsum().apply(np.exp).plot(figsize=(10, 6))
    plt.savefig('img2.png', bbox_inches="tight")
    plt.clf()
    image2 = Image('img2.png', width=200, height=200)
    content_flow.append(image2)
    # Overall returns across the test period for both static and model based strategies
    paragraph_txt = Paragraph('Return seen for just holding stock = ' + str(round(np.exp(test[ticker + '_Return'].sum()), 2)), sample_style_sheet['Heading2'])
    content_flow.append(paragraph_txt)
    paragraph_txt = Paragraph('Return seen using model strategy = ' + str(round(np.exp(test[ticker + '_model_results'].sum()), 2)), sample_style_sheet['Heading2'])
    content_flow.append(paragraph_txt)
    # Calculate the number of trades made if following the model strategy across the testing period
    # Making a trade costs money and so impacts the return of any strategy
    paragraph_txt = Paragraph('Across testing period of ' + str(len(test)) + ' days', sample_style_sheet['Heading2'])
    content_flow.append(paragraph_txt)
    trades_made = (test[ticker + '_model_position'].diff() != 0).sum()
    paragraph_txt = Paragraph('Trades made using strategy = ' + str(trades_made), sample_style_sheet['Heading2'])
    content_flow.append(paragraph_txt)
    # Generate the report
    pdf = SimpleDocTemplate(report_dir + 'Model_Perf_' + ticker + '_' + date + '.pdf')
    pdf.build(content_flow)
    # Tidy up afterwards by removing the images
    os.remove("img1.png")
    os.remove("img2.png")

def challenger_report(model, columns, train_cp, test_cp, stock, date):
    '''
    A function to generate the challenger report
    :param model: the trained challenger model
    :param columns: the columns on which the model was trained
    :param train_cp: the enhanced training datset
    :param test_cp: the enhanced testing dataset
    :param stock: the stock of interest
    :param date: the report execution date
    :return: none
    '''
    import sklearn.tree as tr

    # The list variable containing the report content. Built up as we go along
    content_flow = []
    paragraph_txt = Paragraph("Challenger model : Decision Tree Classifier", sample_style_sheet['Heading1'])
    content_flow.append(paragraph_txt)
    paragraph_txt = Paragraph(stock + " on " + date, sample_style_sheet['Heading2'])
    content_flow.append(paragraph_txt)
    # The challenger model score on the training set
    paragraph_txt = Paragraph('Training set model score = ' + str(round(model.score(train_cp[columns], train_cp[stock + '_Direction_bin']), 2)), sample_style_sheet['Heading3'])
    content_flow.append(paragraph_txt)
    # The challenger model return across the training set
    train_cp[[stock + '_Return', stock + '_model_results']].cumsum().apply(np.exp).plot(figsize=(10, 6))
    plt.savefig('img1.png', bbox_inches="tight")
    plt.clf()
    image1 = Image('img1.png', width=200, height=200)
    content_flow.append(image1)
    # The challenger model score on the testing set
    paragraph_txt = Paragraph('Testing set model score = ' + str(round(model.score(test_cp[columns], test_cp[stock + '_Direction_bin']), 2)), sample_style_sheet['Heading3'])
    content_flow.append(paragraph_txt)
    # The challenger model return across the testing set
    test_cp[[stock + '_Return', stock + '_model_results']].cumsum().apply(np.exp).plot(figsize=(10, 6))
    plt.savefig('img2.png', bbox_inches="tight")
    plt.clf()
    image2 = Image('img2.png', width=200, height=200)
    content_flow.append(image2)
    # Print out the returns over the test period for both static and model based strategies
    paragraph_txt = Paragraph('Return seen for just holding stock = ' + str(round(np.exp(test_cp[stock + '_Return'].sum()), 2)), sample_style_sheet['Heading3'])
    content_flow.append(paragraph_txt)
    paragraph_txt = Paragraph('Return seen using model strategy = ' + str(round(np.exp(test_cp[stock + '_model_results'].sum()), 2)), sample_style_sheet['Heading3'])
    content_flow.append(paragraph_txt)
    # Use the sklearn plot_tree function which generates a pictorial view of the decision tree
    # Allows the user to compare against the production model anchor report
    plt.figure(figsize=(40, 12))
    tr.plot_tree(model, feature_names=columns, class_names=['+1', '-1'], fontsize=8)
    plt.savefig('img3.png', bbox_inches="tight")
    plt.clf()
    # For the decision tree plot, we need to change the page orientation to be landscape to enable it to be seen clearly
    # To do this I have used the following code taken from stackoverflow
    # https://stackoverflow.com/questions/50660395/reportlab-how-to-change-page-orientation/50660701
    pdf = BaseDocTemplate(report_dir + 'Challenger_Report_' + stock + "_" + date + '.pdf', pagesize=A4, rightMargin=25, leftMargin=25, topMargin=25, bottomMargin=25)
    portrait_frame = Frame(pdf.leftMargin, pdf.bottomMargin, pdf.width, pdf.height, id='portrait_frame ')
    landscape_frame = Frame(pdf.leftMargin, pdf.bottomMargin, pdf.height, pdf.width, id='landscape_frame ')
    pdf.addPageTemplates([PageTemplate(id='portrait', frames=portrait_frame), PageTemplate(id='landscape', frames=landscape_frame, pagesize=landscape(A4))])
    content_flow.append(NextPageTemplate('landscape'))
    image3 = Image('img3.png', height=400, width=700)
    content_flow.append(image3)
    # Build the report
    pdf.build(content_flow)
    # Tidy up afterwards by removing the images
    os.remove("img1.png")
    os.remove("img2.png")
    os.remove("img3.png")

def alt_models_report(train, test, portfolio, date):
    '''
    A function to generate benchmark results for the same dataset using a number of different classifiers
    :param train: the training dataset
    :param test: the testing dataset
    :param portfolio: the portfolio of stocks
    :param date: the execution date
    :return: none
    '''
    from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.gaussian_process.kernels import RBF
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.svm import SVC
    from __main__ import train_model

    # Specifies the classifiers to be included
    # I have used the list and configuration as specified in the sklearn documentation
    # https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
    # I have excluded the MLP Classifier as this is our production model and the Decision Tree Classifier which is our challenger
    # I have added to the sklearn list the Gradient Boosting Classifier which came up in the literature review
    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025, random_state=42),
        SVC(gamma=2, C=1, random_state=42),
        GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, random_state=42),
        AdaBoostClassifier(algorithm="SAMME", random_state=42),
        GaussianNB(),
        GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=42)
    ]
    # A report is generated for each stock in the portfolio for each alternative model
    for stock in portfolio:
        # The list variable containing the report content. Built up as we go along
        content_flow = []
        paragraph_txt = Paragraph("Alternative model performance report", sample_style_sheet['Heading1'])
        content_flow.append(paragraph_txt)
        paragraph_txt = Paragraph("Testing dataset for " + stock + " on " + date, sample_style_sheet['Heading2'])
        content_flow.append(paragraph_txt)
        # Add the details for each classifier in turn
        for classifier in classifiers:
            paragraph_txt = Paragraph("Model : " + str(classifier), sample_style_sheet['Heading3'])
            content_flow.append(paragraph_txt)
            # Train the model
            model_trained, columns_trained = train_model(classifier, train, stock)
            # Calculate the model score when run on the training dataset
            paragraph_txt = Paragraph('Model score = ' + str(round(model_trained.score(test[columns_trained], test[stock + '_Direction_bin']), 2)), sample_style_sheet['Heading3'])
            content_flow.append(paragraph_txt)
        # Build the report
        pdf = SimpleDocTemplate(report_dir + 'Alt_Model_Report_' + stock + "_" + date + '.pdf')
        pdf.build(content_flow)