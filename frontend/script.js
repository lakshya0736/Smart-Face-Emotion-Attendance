document.getElementById("markBtn").addEventListener("click", async () => {
    const btn = document.getElementById("markBtn");
    btn.disabled = true;
    btn.innerText = "Marking Attendance...";

    try {
        const response = await fetch("/mark_attendance", { method: "POST" });
        const data = await response.json();

        alert(data.message);
    } catch (err) {
        alert("Error marking attendance. Please try again.");
    }

    btn.disabled = false;
    btn.innerText = "Mark Attendance";
});
