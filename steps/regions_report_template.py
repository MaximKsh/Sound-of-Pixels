sbr_html = """
<html>
  <head>
    <title></title>
    <meta content="">
    <style></style>
  </head>
  <body>

    <table style="width:50%">
        <tr>
            <th><img src="masks-grid.jpg" height=448 width=448/></th>
            <th><img class="frame" src="frame.jpg"  height=448 width=448/></th>
        </tr>
    </table>

    <script>
    function offset(el) {
	    var rect = el.getBoundingClientRect(),
	    scrollLeft = window.pageXOffset || document.documentElement.scrollLeft,
	    scrollTop = window.pageYOffset || document.documentElement.scrollTop;
	    return { top: rect.top + scrollTop, left: rect.left + scrollLeft }
	}

    function printMousePos(event) {
        var frame = document.getElementsByClassName("frame")[0];
        var pos = offset(frame)
        var x = Math.floor((event.clientX - pos["left"]) / 32);
        var y = Math.floor((event.clientY - pos["top"]) / 32);

        console.log("clientX: " + (event.clientX - pos["left"]) + " - clientY: " + (event.clientY - pos["top"]));

        sound_filename = "grid/" + y + "x" + x + "-pred.wav";
        console.log(sound_filename);
        var audio = new Audio(sound_filename);
        audio.play();
    }

    var frame = document.getElementsByClassName("frame");
    frame[0].addEventListener("click", printMousePos);
    </script>

  </body>
</html>
"""