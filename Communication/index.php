<html>
<head>
<title> User Data </title>
<style>
  table {
      font-family: "Trebuchet MS", Arial, Helvetica, sans-serif;
      border-collapse: collapse;
      width: 100%;
  }

  td, th {
      border: 1px solid #ddd;
      padding: 8px;
  }

  tr:nth-child(even){background-color: #f2f2f2;}

  tr:hover {background-color: #ddd;}

  th {
      padding-top: 12px;
      padding-bottom: 12px;
      text-align: left;
      background-color: #4CAF50;
      color: white;
  }
</style>
</head>
<body>

<?php
// use our connect php script to connect to vignette_db
include('connect-mysql.php');

// select all columns in the user data table
$sqlget = "SELECT * FROM user_data";
$sqldata = mysqli_query($dbcon, $sqlget) or die ('error fetching data');

echo "<table>";
echo "<tr><th>Time</th><th>Item</th><th>Frequency</th></tr>";

// fetch each row as an associative array mapped to each column header
while($row = mysqli_fetch_array($sqldata, MYSQLI_ASSOC)) {
  echo "<tr><td>";
  echo $row['Time'];
  echo "</td><td>";
  echo $row['Item'];
  echo "</td><td>";
  echo $row['Frequency'];
  echo "</td></tr>";
}

// free the data that was queries and close the connection to the db
mysqli_free_result($sqldata);
mysqli_close($dbcon);
?>

</body>
</html>