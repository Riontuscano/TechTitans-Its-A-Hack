const data = [
    { time: "11:30:49", code: "00.000.0002", location: "Virar, Mumbai" },
    { time: "12:15:30", code: "00.000.0003", location: "Andheri, Mumbai" },
    { time: "13:45:10", code: "00.000.0004", location: "Bandra, Mumbai" }
];

const tbody = document.getElementById('dataBody');

data.forEach(item => {

    const tr = document.createElement('tr');
    
    const tdTime = document.createElement('td');
    tdTime.className = 'time';
    tdTime.innerHTML = `<p>${item.time}</p>`;
    
    const tdCode = document.createElement('td');
    tdCode.className = 'location';
    tdCode.innerHTML = `<p>${item.code}</p>`;
    
    const tdLocation = document.createElement('td');
    tdLocation.className = 'location';
    tdLocation.innerHTML = `<p>${item.location}</p>`;
    

    tr.appendChild(tdTime);
    tr.appendChild(tdCode);
    tr.appendChild(tdLocation);

    tbody.appendChild(tr);
});
