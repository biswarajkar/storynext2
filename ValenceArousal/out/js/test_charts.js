$(document).ready(function() {
    var pos_test_all = $('#pos_test_all');
    var pos_test_doc = $('#pos_test_doc');


    var setTooltip = function(prefix, valence, arousal, sentence) {
        var currentValenceTooltip = $('#' + prefix + '_valence_tooltip');
        var currentArousalTooltip = $('#' + prefix + '_arousal_tooltip');
        var currentSentenceTooltip = $('#' + prefix + '_sentence_tooltip');

        currentValenceTooltip.text(valence);
        currentArousalTooltip.text(arousal);
        currentSentenceTooltip.text(sentence);
    };

    var generateChartOptions = function(graph_label, data, prefix, afterBody_cb, label_cb) {
        return {
            type: 'scatter',
            data: {
                datasets: [{
                    label: graph_label,
                    data: data
                }]
            },
            lineAtIndex: [5],
            options: {
                tooltips: {
                    mode: 'point',
                    bodySpacing: 8,
                    callbacks: {
                        afterBody: function(tooltipItems, data) {
                            dataObject = data.datasets[0].data[tooltipItems[0].index];
                            // display valence, arousal, and sentence on browser
                            setTooltip(prefix, dataObject.y, dataObject.x, dataObject.sentence);
                            maxLength = 70;
                            if (dataObject.sentence.length > maxLength) {
                                return dataObject.sentence.substring(0,maxLength - 2) + '...';
                            } else {
                                return dataObject.sentence;
                            }
                        },
                        label: function(tooltipItem, data) {
                            dataObject = data.datasets[0].data[tooltipItem.index];
                            var string = "Arousal: " + dataObject.x +
                            ", Valence: " + dataObject.y
                            return string
                        }
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
                }
            }
        };
    };

    var after_body_cb_for_all_sentences = function(tooltipItems, data) {
        dataObject = data.datasets[0].data[tooltipItems[0].index];
        // display valence, arousal, and sentence on browser
        setTooltip(prefix, dataObject.y, dataObject.x, dataObject.sentence);
        maxLength = 70;
        if (dataObject.sentence.length > maxLength) {
            return dataObject.sentence.substring(0,maxLength - 2) + '...';
        } else {
            return dataObject.sentence;
        }
    };

    var label_cb_for_all_sentences = function(tooltipItem, data) {
        dataObject = data.datasets[0].data[tooltipItem.index];
        var string = "Arousal: " + dataObject.x +
        ", Valence: " + dataObject.y
        return string
    };

    var after_body_cb_for_per_doc = function(tooltipItems, data) {

    };

    var label_cb_for_per_doc = function(tooltipItem, data) {

    };

    var pos_test_all = new Chart(pos_test_all,
        generateChartOptions('Positive Test -- All sentences', v_and_a_pos, 'pos_test_all',
            after_body_cb_for_all_sentences, label_cb_for_all_sentences));

    var pos_test_doc = new Chart(pos_test_doc,
        generateChartOptions('Positive Test -- By Document', v_and_a_per_doc_pos, 'pos_test_doc',
            after_body_cb_for_per_doc, label_cb_for_per_doc));


});