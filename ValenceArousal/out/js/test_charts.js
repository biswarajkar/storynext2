$(document).ready(function() {
    var ctx = $('#testChart')
    var currentValenceTooltip = $('#valence_tooltip');
    var currentArousalTooltip = $('#arousal_tooltip');
    var currentSentenceTooltip = $('#sentence_tooltip');

    var setTooltip = function(valence, arousal, sentence) {
        currentValenceTooltip.text(valence);
        currentArousalTooltip.text(arousal);
        currentSentenceTooltip.text(sentence);
    }

    var scatterChart = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [{

                label: 'Valence and Arousal',
                data: v_and_a
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
                        setTooltip(dataObject.y, dataObject.x, dataObject.sentence);
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
                        borderWidth: 2,
                        label: {
                            backgroundColor: "black",
                            content: "valence",
                            position: "top",
                            enabled: true
                        }
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
                            content: "arousal",
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

    });
});