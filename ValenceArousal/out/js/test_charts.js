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

    var generateChartOptions = function(ctx, graph_label, data, prefix, afterBody_cb, label_cb) {


        return {
            type: 'scatter',
            data: {
                datasets: [{
                    label: graph_label,
                    data: data,
                    borderColor:               gradientStroke,
                    pointBorderColor:          '#999999',
                    pointBackgroundColor:      gradientStroke,
                    pointHoverBackgroundColor: gradientStroke,
                    pointHoverBorderColor:     gradientStroke
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
        var string = "Arousal: " + dataObject.y +
        ", Valence: " + dataObject.x
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
        var string = "Arousal: " + dataObject.y +
        ", Valence: " + dataObject.x
        return string
    };

    var gradientStroke = document.getElementById('pos_test_all').getContext("2d").createLinearGradient(30, 0, 520, 0);
    gradientStroke.addColorStop(0, "#ff0000");
    gradientStroke.addColorStop(.5, "#ffffff")
    gradientStroke.addColorStop(1, "#0008ff");


    var pos_test_all = new Chart(pos_test_all,
        generateChartOptions(pos_test_all, 'Positive Test -- All sentences', v_and_a_pos, 'pos_test_all',
            after_body_cb_for_all_sentences('pos_test_all'), label_cb_for_all_sentences));

    var pos_test_doc = new Chart(pos_test_doc,
        generateChartOptions(pos_test_doc, 'Positive Test -- By Document', v_and_a_per_doc_pos, 'pos_test_doc',
            after_body_cb_for_per_doc('pos_test_doc'), label_cb_for_per_doc));

    var pos_test_all = new Chart(neg_test_all,
        generateChartOptions(neg_test_all, 'Negative Test -- All sentences', v_and_a_neg, 'neg_test_all',
            after_body_cb_for_all_sentences('neg_test_all'), label_cb_for_all_sentences));

    var pos_test_doc = new Chart(neg_test_doc,
        generateChartOptions(neg_test_doc, 'Negative Test -- By Document', v_and_a_per_doc_neg, 'neg_test_doc',
            after_body_cb_for_per_doc('neg_test_doc'), label_cb_for_per_doc));

});