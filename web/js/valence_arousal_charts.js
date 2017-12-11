$(document).ready(function() {

    var per_sentence_sentiment_ref = $('#per_sentence_sentiment');
    var per_word_sentiment_ref = $('#per_word_sentiment');
    var sentiment_over_time_ref = $('#sentiment-over-time-chart');


    var setTooltip = function(prefix, valence, arousal, sentence, domain) {
        var currentValenceTooltip = $('#' + prefix + '_valence_tooltip');
        var currentArousalTooltip = $('#' + prefix + '_arousal_tooltip');
        var currentSentenceTooltip = $('#' + prefix + '_sentence_tooltip');

        currentValenceTooltip.text(valence);
        currentArousalTooltip.text(arousal);
        currentSentenceTooltip.text(domain + ' ' + sentence);
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
                return dataObject.domain + ' ' + dataObject.sentence.substring(0,maxLength - 2) + '...';
            } else {
                return dataObject.domain + ' ' + dataObject.sentence;
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
    //    ajax_request('js/mean_valence_and_arousals_pos.js').then(function(va_pos) {
    //        per_sentence_sentiment = new Chart(per_sentence_sentiment,
    //            generateChartOptions(per_sentence_sentiment, 'Positive Test -- All sentences', va_pos.v_and_a_pos, 'per_sentence_sentiment',
    //                after_body_cb_for_all_sentences('per_sentence_sentiment'), label_cb_for_all_sentences, per_sentence_sentiment_colors));
    //    }).then(function() {
    //        per_word_sentiment = new Chart(per_word_sentiment,
    //            generateChartOptions(per_word_sentiment, 'Positive Test -- By Document', v_and_a_per_doc_pos, 'per_word_sentiment',
    //                after_body_cb_for_per_doc('per_word_sentiment'), label_cb_for_per_doc, per_word_sentiment_colors));
    //    });
    };

    var update_charts = function(text) {
        //var addData = function(chart, labels, data) {
        //    labels.forEach(label => chart.data.labels.push(label));
        //    chart.data.datasets.forEach((dataset) => {
        //        dataset.data.push(data);
        //    });
        //    chart.update();
        //};
//
        //var removeAllData = function(chart) {
        //    chart.data.labels = [];
        //    chart.data.datasets.forEach((dataset) => {
        //        dataset.data = [];
        //    });
        //    chart.update();
        //};
        //var data, label;

        /// To update charts, you have a list of data and a list of labels. Those are then paired up and added to the chart
        //ajax_request('js/mean_valence_and_arousals_pos.js').then(function(va_pos) {
        //    removeData(per_sentence_sentiment);
        //    addData(per_sentence_sentiment, )
        //    per_sentence_sentiment = new Chart(per_sentence_sentiment,
        //        generateChartOptions(per_sentence_sentiment, 'Positive Test -- All sentences', va_pos.v_and_a_pos, 'per_sentence_sentiment',
        //            after_body_cb_for_all_sentences('per_sentence_sentiment'), label_cb_for_all_sentences, per_sentence_sentiment_colors));
        //}).then(function() {
        //    per_word_sentiment = new Chart(per_word_sentiment,
        //        generateChartOptions(per_word_sentiment, 'Positive Test -- By Document', v_and_a_per_doc_pos, 'per_word_sentiment',
        //            after_body_cb_for_per_doc('per_word_sentiment'), label_cb_for_per_doc, per_word_sentiment_colors));
        //});
    };



   per_sentence_sentiment = new Chart(per_sentence_sentiment_ref,
        generateChartOptions(per_sentence_sentiment, 'Sentence sentiment', v_and_a_pos, 'per_sentence_sentiment',
            after_body_cb_for_all_sentences('per_sentence_sentiment'), label_cb_for_all_sentences, per_sentence_sentiment_colors));

   per_word_sentiment = new Chart(per_word_sentiment_ref,
        generateChartOptions(per_word_sentiment, 'Word sentiment', v_and_a_pos_per_word, 'per_word_sentiment',
            after_body_cb_for_per_word('per_word_sentiment'), label_cb_for_per_word, per_word_sentiment_colors));

   // Construct line graph from available data
   var v_and_a_pos_over_time_labels = []
   var v_and_a_pos_over_time_data = []
   var max_length = Math.min(100, v_and_a_pos.length);
   for (var i = 0; i < max_length; i++) {
        var datum = v_and_a_pos[i];
        v_and_a_pos_over_time_labels.push(i);
        v_and_a_pos_over_time_data.push(datum.x);
   }
   var sentiment_over_time_colors = [];
   sentiment_over_time = new Chart(sentiment_over_time_ref, {
        type: 'line',
        data: {
            labels: v_and_a_pos_over_time_labels,
            datasets: [{
                label: "Sentiment of each sentence, by index",
                data: v_and_a_pos_over_time_data,
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
    }

    var refresh_charts = function() {
        // Set chart colors
        scatterplot_chart_colors(per_sentence_sentiment, per_sentence_sentiment_colors);
        scatterplot_chart_colors(per_word_sentiment, per_word_sentiment_colors);
        line_chart_colors(sentiment_over_time, sentiment_over_time_colors);
    };

    init_charts();
    refresh_charts();

});