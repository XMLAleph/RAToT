function showpage(page){
    document.getElementById("page1").style.display="none";
    document.getElementById("page2").style.display="none";
    document.getElementById("page3").style.display="none";
    document.getElementById("page4").style.display="none";
    
    document.getElementById("page"+page).style.display="block";
}