$(document).ready(function() {

    var pos_test_all = $('#pos_test_all');
    var pos_test_doc = $('#pos_test_doc');
    var neg_test_all = $('#neg_test_all');
    var neg_test_doc = $('#neg_test_doc');


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
                    mode: 'point',
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
            setTooltip(prefix, dataObject.y, dataObject.x, dataObject.sentence, dataObject.domain);
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
        return string
    };

    var after_body_cb_for_per_doc = function(prefix) {
        return function(tooltipItems, data) {
            setTooltip(prefix, dataObject.y, dataObject.x, dataObject.domain, '');
            dataObject = data.datasets[0].data[tooltipItems[0].index];
            return dataObject.domain
        };
    };

    var label_cb_for_per_doc = function(tooltipItem, data) {
        dataObject = data.datasets[0].data[tooltipItem.index];
        var string = "Arousal: " + (Math.round(dataObject.y*100)/100) +
        ", Valence: " + (Math.round(dataObject.x*100)/100)
        return string
    };

    var gradientStroke = document.getElementById('pos_test_all').getContext("2d").createLinearGradient(30, 0, 520, 0);
    gradientStroke.addColorStop(0, "#ff0000");
    gradientStroke.addColorStop(.5, "#ffffff")
    gradientStroke.addColorStop(1, "#0008ff");


    var pos_test_all_colors = [];
    var pos_test_doc_colors = [];
    var pos_test_all;
    var pos_test_doc;

    var init_charts = function() {
    //    ajax_request('js/mean_valence_and_arousals_pos.js').then(function(va_pos) {
    //        pos_test_all = new Chart(pos_test_all,
    //            generateChartOptions(pos_test_all, 'Positive Test -- All sentences', va_pos.v_and_a_pos, 'pos_test_all',
    //                after_body_cb_for_all_sentences('pos_test_all'), label_cb_for_all_sentences, pos_test_all_colors));
    //    }).then(function() {
    //        pos_test_doc = new Chart(pos_test_doc,
    //            generateChartOptions(pos_test_doc, 'Positive Test -- By Document', v_and_a_per_doc_pos, 'pos_test_doc',
    //                after_body_cb_for_per_doc('pos_test_doc'), label_cb_for_per_doc, pos_test_doc_colors));
    //    });
    };



   pos_test_all = new Chart(pos_test_all,
        generateChartOptions(pos_test_all, 'Positive Test -- All sentences', v_and_a_pos, 'pos_test_all',
            after_body_cb_for_all_sentences('pos_test_all'), label_cb_for_all_sentences, pos_test_all_colors));

   pos_test_doc = new Chart(pos_test_doc,
        generateChartOptions(pos_test_doc, 'Positive Test -- By Document', v_and_a_per_doc_pos, 'pos_test_doc',
            after_body_cb_for_per_doc('pos_test_doc'), label_cb_for_per_doc, pos_test_doc_colors));


    var chart_colors = function(chart, colors) {
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

    var refresh_charts = function() {
        // Set chart colors
        chart_colors(pos_test_all, pos_test_all_colors);
        chart_colors(pos_test_doc, pos_test_doc_colors);
    };

    init_charts();
    refresh_charts();

});