function addList() {
  const li = document.createElement("p");
  const ratioArr = document.getElementsByName("ratio");
  ratioID = ratioArr.length + 1;
  li.innerHTML = `<div class="btn btn-light btn-icon-split">
                  <span class="icon text-gray-600"> 종목 </span>
                  <input type="text" name="ticker" id="ticker_${parseInt(
                    ratioID
                  )}" required />
                </div>
                <div class="btn btn-light btn-icon-split">
                  <input
                    type="number"
                    name="ratio"
                    id="ratio_${parseInt(ratioID)}"
                    min="0"
                    max="100"
                    required
                    onkeyup="checkInputValue(this)"
                    onclick="checkInputValue(this)"
                  />
                  <span class="icon text-gray-600"> % </span>
                </div>`;
  document.getElementById("assets").appendChild(li);
}

function removeItem() {
  const assets = document.getElementById("assets");
  const items = assets.getElementsByTagName("p");
  if (items.length > 1) {
    items[items.length - 1].remove();
  }
  sumRatio();
}

function checkInputValue(obj) {
  val = obj.value;
  if (val < 0 || val > 100) {
    obj.value = null;
    alert("0에서 100 사이 숫자만 입력 가능합니다!");
  }
  if (sumRatio()) {
    obj.value = null;
    alert("총 합이 100% 이하여야 합니다!");
    sumRatio();
  }
}

function checkMoneyValue(obj) {
  val = obj.value;
  if (val < 0) {
    obj.value = null;
    alert("초기 자산은 음수일 수 없습니다!");
  }
}

function sumRatio() {
  const ratioList = document.getElementsByName("ratio");
  const total = document.getElementById("percent-number");
  let result = 0;
  for (const ratio of ratioList.values()) {
    val = ratio.value;
    if (val == "") {
      val = 0;
    }
    result += parseFloat(val);
  }
  if (result > 100) {
    return true;
  }
  total.innerHTML = result;
}

function validateForm() {
  const total = document.getElementById("percent-number");
  const startYear = document.getElementsByName("startYear")[0].value;
  const startMonth = document.getElementsByName("startMonth")[0].value;
  const endYear = document.getElementsByName("endYear")[0].value;
  const endMonth = document.getElementsByName("endMonth")[0].value;

  if (parseFloat(total.innerHTML) != 100) {
    alert("총 합이 100% 이어야 합니다!");
    return false;
  }

  if (startYear > endYear) {
    alert("시작년도가 종료년도보다 클 수 없습니다.");
    return false;
  }

  if (startYear === endYear && startMonth >= endMonth) {
    alert(
      "시작년도가 종료년도와 같을 때, startMonth가 endMonth보다 클 수 없습니다."
    );
    return false;
  }
}
