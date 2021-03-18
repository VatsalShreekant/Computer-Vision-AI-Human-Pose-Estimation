// let yogaPose = $(".pose-description").text();
// alert(yogaPose);

let yogaPose = "";

$(".mountain-img").on('click', function(){
    yogaPose = $("#Mountain").attr("id");
    localStorage.setItem('SelectedPose', yogaPose);
});

$(".tree-img").on('click', function(){
    yogaPose = $("#Tree").attr("id");
    localStorage.setItem('SelectedPose', yogaPose);
});

$(".goddess-img").on('click', function(){
    yogaPose = $("#Goddess").attr("id");
    localStorage.setItem('SelectedPose', yogaPose);
});

$(".warrior2-img").on('click', function(){
    yogaPose = $("#Warrior-2").attr("id");
    localStorage.setItem('SelectedPose', yogaPose);
});