$(document).ready(function() {

    var per_sentence_sentiment_ref = $('#per_sentence_sentiment');
    var per_word_sentiment_ref = $('#per_word_sentiment');
    var sentiment_over_time_ref = $('#sentiment-over-time-chart');

    $(".go").on("click", function () {
        press_go();
    });

    var press_go = function () {
        $.ajax({
            url: "cgi-bin/testAjax.cgi",
            type: "POST",
            data: {content: $("#text-input").val(), bar: 'bar'},
            success: function (response) {
                //$("#testAjax").html(response);
                // alert(response);
                update_charts(response);
                //console.log(results);
            }
        });
    };


    var generateChartOptions = function(ctx, graph_label, data, prefix, afterBody_cb, label_cb, colors) {


        return {
            type: 'scatter',
            data: {
                datasets: [{
                    label: graph_label,
                    data: data,
                    borderColor:               colors,
                    pointBorderColor:          '#999999',
                    pointBackgroundColor:      colors,
                    pointHoverBackgroundColor: colors,
                    pointHoverBorderColor:     colors
                }]
            },
            lineAtIndex: [5],
            options: {
                tooltips: {
                    mode: 'nearest',
                    bodySpacing: 8,
                    callbacks: {
                        afterBody: afterBody_cb,
                        label: label_cb
                    }
                },
                annotation: {
                    events: ["click"],
                    annotations: [
                          {
                            drawTime: "afterDatasetsDraw",
                            id: "vline",
                            type: "line",
                            mode: "vertical",
                            scaleID: "x-axis-0",
                            value: 5,
                            borderColor: "black",
                            borderDash: [2,2],
                            borderWidth: 2/*,
                            label: {
                                backgroundColor: "black",
                                content: "arousal",
                                position: "top",
                                enabled: true
                            }*/
                          },
                          {
                            drawTime: "afterDatasetsDraw",
                            id: "hline",
                            type: "line",
                            mode: "horizontal",
                            scaleID: "y-axis-0",
                            value: 5,
                            borderColor: "black",
                            borderDash: [2,2],
                            borderWidth: 2,
                            label: {
                                backgroundColor: "black",
                                content: "valence",
                                position: "right",
                                enabled: true
                            }
                          }
                    ]
                },
                responsive: true,
                scales: {
                    yAxes: [{
                        id: 'y-axis-0',
                        display: true,
                        ticks: {
                            suggestedMin: 1,
                            max: 9
                        }
                    }],

                    xAxes: [{
                        id: 'x-axis-0',
                        display: true,
                        ticks: {
                            suggestedMin: 1,
                            max: 9
                        }
                    }]
                }/*,
                pan: {
                    // Boolean to enable panning
                    enabled: true,
                    // Panning directions. Remove the appropriate direction to disable
                    // Eg. 'y' would only allow panning in the y direction
                    mode: 'xy'
                },
                // Container for zoom options
                zoom: {
                    // Boolean to enable zooming
                    enabled: true,

                    // Zooming directions. Remove the appropriate direction to disable
                    // Eg. 'y' would only allow zooming in the y direction
                    mode: 'xy',
                }*/
            }
        };
    };

    var after_body_cb_for_all_sentences = function(prefix) {
        return function(tooltipItems, data) {
            dataObject = data.datasets[0].data[tooltipItems[0].index];
            // display valence, arousal, and sentence on browser
            //setTooltip(prefix, dataObject.y, dataObject.x, dataObject.sentence, dataObject.domain);
            maxLength = 60;
            if (dataObject.sentence.length > maxLength) {
                return dataObject.sentence.substring(0,maxLength - 2) + '...';
            } else {
                return dataObject.sentence;
            }
        };
    };

    var label_cb_for_all_sentences = function(tooltipItem, data) {
        dataObject = data.datasets[0].data[tooltipItem.index];
        var string = "Arousal: " + (Math.round(dataObject.y*100)/100) +
        ", Valence: " + (Math.round(dataObject.x*100)/100)
        return string;
    };

    var after_body_cb_for_per_word = function(prefix) {
        return function(tooltipItems, data) {
            //setTooltip(prefix, dataObject.y, dataObject.x, dataObject.domain, '');
            dataObject = data.datasets[0].data[tooltipItems[0].index];
            return dataObject.word;
        };
    };

    var label_cb_for_per_word = function(tooltipItem, data) {
        dataObject = data.datasets[0].data[tooltipItem.index];
        var string = "Arousal: " + (Math.round(dataObject.y*100)/100) +
        ", Valence: " + (Math.round(dataObject.x*100)/100)
        return string;
    };

    var per_sentence_sentiment_colors = [];
    var per_word_sentiment_colors = [];
    var per_sentence_sentiment;
    var per_word_sentiment;
    var sentiment_over_time;

    var init_charts = function() {
        press_go();
    };

    // data = {'sentiment_per_word' : ...,
    //          'sentiment_per_sentence': ...,
    //          'sentiment_over_time': ...}
    var update_charts = function(python_response) {
        var addData = function(chart, labels, data) {
           labels.forEach(label => chart.data.labels.push(label));
           chart.data.datasets.forEach((dataset) => {
               data.forEach(d => dataset.data.push(d));
           });
           chart.update();
        };

        var addScatterplotData_Sentence = function (chart, labels, data, texts) {
            chart.data.datasets.forEach((dataset) => {
                for (var i = 0; i < data.length; i++) {
                    var d = {
                        'x': labels[i],
                        'y': data[i],
                        'sentence': texts[i]
                    };
                    dataset.data.push(d);
                }
            });
        };

        var addScatterplotData_Word = function (chart, labels, data, texts) {
            chart.data.datasets.forEach((dataset) => {
                for (var i = 0; i < data.length; i++) {
                    var d = {
                        'x': labels[i],
                        'y': data[i],
                        'word': texts[i]
                    };
                    dataset.data.push(d);
                }
            });
        };

        var removeAllData = function(chart) {
           chart.data.labels = [];
           chart.data.datasets.forEach((dataset) => {
               dataset.data = [];
           });
           chart.update();
        };

        removeAllData(sentiment_over_time);
        removeAllData(per_sentence_sentiment);
        removeAllData(per_word_sentiment);

        addData(sentiment_over_time, python_response.sentiment_over_time.labels, python_response.sentiment_over_time.data);
        addScatterplotData_Sentence(per_sentence_sentiment, python_response.sentiment_per_sentence.labels,
            python_response.sentiment_per_sentence.data,
            python_response.sentiment_per_sentence.texts
        );
        addScatterplotData_Word(per_word_sentiment, python_response.sentiment_per_word.labels,
            python_response.sentiment_per_word.data,
            python_response.sentiment_per_word.texts
        );

        refresh_charts();
    };



   per_sentence_sentiment = new Chart(per_sentence_sentiment_ref,
        generateChartOptions(per_sentence_sentiment, 'Sentence sentiment', [], 'per_sentence_sentiment',
            after_body_cb_for_all_sentences('per_sentence_sentiment'), label_cb_for_all_sentences, per_sentence_sentiment_colors));

   per_word_sentiment = new Chart(per_word_sentiment_ref,
        generateChartOptions(per_word_sentiment, 'Word sentiment', [], 'per_word_sentiment',
            after_body_cb_for_per_word('per_word_sentiment'), label_cb_for_per_word, per_word_sentiment_colors));

   // Construct line graph from available data
   var sentiment_over_time_colors = [];
   sentiment_over_time = new Chart(sentiment_over_time_ref, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: "Sentiment of each sentence, by index",
                data: [],
                borderColor:               sentiment_over_time_colors,
                pointBorderColor:          '#000',
                pointBackgroundColor:      sentiment_over_time_colors,
                pointHoverBackgroundColor: sentiment_over_time_colors,
                pointHoverBorderColor:     sentiment_over_time_colors,
                fill: false
            }]
        },
        options: {
                maintainAspectRatio: false,
                tooltips: {
                    mode: 'nearest',
                    bodySpacing: 8,
                    callbacks: {
                        title: function(tooltipItems, data) {
                            return "Sentence " + tooltipItems[0].xLabel;
                        },
                        afterBody: function(tooltipItems, data) {
                            dataObject = data.datasets[0].data[tooltipItems[0].index];
                            return "";
                        },
                        label: function(tooltipItem, data) {
                            valence = data.datasets[0].data[tooltipItem.index];
                            var string = "Valence: " + (Math.round(valence*100)/100)
                            return string;
                        }
                    }
                },
                annotations: [
                    {
                      drawTime: "afterDatasetsDraw",
                      id: "hline",
                      type: "line",
                      mode: "horizontal",
                      scaleID: "y-axis-0",
                      value: 5,
                      borderColor: "black",
                      borderDash: [2,2],
                      borderWidth: 2,
                      label: {
                          backgroundColor: "black",
                          content: "valence",
                          position: "right",
                          enabled: true
                      }
                    }
                ],
                responsive: true,
                scales: {
                    yAxes: [{
                        id: 'y-axis-0',
                        display: true,
                        ticks: {
                            suggestedMin: 1,
                            max: 9
                        }
                    }],
                    xAxes: [{
                        id: 'x-axis-0',
                        display: true,
                        ticks: {
                            autoSkip: true,
                            maxTicksLimit: 10
                        }
                    }]
                }
            }
   });


    var scatterplot_chart_colors = function(chart, colors) {
        for (i = 0; i < chart.data.datasets[0].data.length; i++) {
            var c = "";
            var valence = chart.data.datasets[0].data[i].x;
            if (valence > 5) {
                var b = 255;
                var r_and_g = Math.floor(255 - (255 * ((valence - 5) / 4)));
                c = "rgb("+r_and_g+","+r_and_g+"," + b + ")"
            } else {
                var r = 255;
                var g_and_b = Math.floor(255 * ((valence - 1) / 4));
                c = "rgb("+r+","+g_and_b+"," + g_and_b + ")"
            }
            colors.push(c);
        }
        chart.update()
    };

    var line_chart_colors = function(chart, colors) {
        for (i = 0; i < chart.data.datasets[0].data.length; i++) {
            var c = "";
            var valence = chart.data.datasets[0].data[i];
            if (valence > 5) {
                var b = 255;
                var r_and_g = Math.floor(255 - (255 * ((valence - 5) / 4)));
                c = "rgb("+r_and_g+","+r_and_g+"," + b + ")"
            } else {
                var r = 255;
                var g_and_b = Math.floor(255 * ((valence - 1) / 4));
                c = "rgb("+r+","+g_and_b+"," + g_and_b + ")"
            }
            colors.push(c);
        }
        chart.update()
    };

    var refresh_charts = function() {
        // Set chart colors
        per_sentence_sentiment_colors.length = 0;
        per_word_sentiment_colors.length = 0;
        sentiment_over_time_colors.length = 0;
        scatterplot_chart_colors(per_sentence_sentiment, per_sentence_sentiment_colors);
        scatterplot_chart_colors(per_word_sentiment, per_word_sentiment_colors);
        line_chart_colors(sentiment_over_time, sentiment_over_time_colors);
    };

    init_charts();
});