var svg;
var Y;

var zoomListener;


function drawEmbedding() {
    $("#embed").empty();
    var div = d3.select("#embed");

    svg = div.append("svg") // svg is global
        .attr("width", canvas_width)
        .attr("height", canvas_height)
        .attr("style", "border:solid 1px");

    var g = svg.selectAll(".b")
      .data(data)
      .enter().append("g")
      .attr("class", "u");

    g.append("circle")
        .attr("r", 5)
        .attr("fill", function(d){
             col_name = d3.rgb(255*d.color[0], 255*d.color[1], 255*d.color[2]);
             return col_name;})
        .attr("stroke", "black")
        .attr("opacity", .4)
        .on("mouseover", point_mouseover)
        .on("touchstart", point_mouseover)
        .on("mouseout", reset_highlight)
        .on("touchend", reset_highlight);


    zoomListener = d3.behavior.zoom()
      .scaleExtent([0.0001, 500])
      .translate([canvas_width / 2, canvas_height / 2])
      .scale(ss)
      .on("zoom", zoomHandler);
    zoomListener(svg);

    updateEmbedding();
}


var line = d3.svg.line().interpolate("linear").x(function(d) {return d[0]*ss + tx}).y(function(d) {return d[1]*ss+ty});

function point_mouseover(point) {
    $("#BeforeTxt").text(point.before.join(' '));
    $("#AfterTxt").text(point.after.join(' '));
    $("#Label").text(point.label);
    $("#pointDataBrush").show();
    $("#instructions").hide();
}

function reset_highlight(d) {
    $("#pointDataBrush").hide();
    $("#instructions").show();
    svg.selectAll('.line').remove();
}

function updateEmbedding() {
  svg.selectAll('.u')
    .attr("transform", function(d) {return "translate(" + ((d.xy[0]*ss + tx)) + "," + ((d.xy[1]*ss + ty)) + ")"; });
  svg.selectAll(".line").attr("d", line);
}

var tx;
var ty;
var ss=2;
function zoomHandler() {
  tx = d3.event.translate[0];
  ty = d3.event.translate[1];
  ss = d3.event.scale;
  updateEmbedding();
}

var data;
function load(json_path) {
	$("#embed").html('<h2><span class="glyphicon glyphicon-refresh glyphicon-refresh-animate"></span> Loading, please wait...</h2>');
    $.getJSON(json_path, function(json) {
        data = json//.slice(5000, 10000);
        drawEmbedding();
      });
}

$(window).load(function() {
    canvas_width = window.innerWidth - 2;
    canvas_height = window.innerHeight - 170;
    tx = canvas_width / 2;
    ty = canvas_height / 2;
    $("#pointDataBrush").hide();
    if (window.location.hash.length > 0) {
        load(window.location.hash.substr(1));
    }
 });

 $(window).resize(function() {
    canvas_width = window.innerWidth - 2;
    canvas_height = window.innerHeight - 170;
    drawEmbedding();
 })

