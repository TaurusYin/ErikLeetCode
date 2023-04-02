"""

import java.io.*;
import java.util.*;
import java.text.*;
import java.math.*;
import java.util.regex.*;


public class Solution {

//give a list of cities names, x coor of cities, y corr of cities, and a city to look for,
// find nearest city of the city having same x or y coor as the city
//if nearest city more than one, return the city name with largest alphabatical value

//put city with same x  into a hashmap x: list of cities with x coor the same [x,y, index]
//sort the list of cities in the mapper
//for each city to find its neibor
    //generate a min dianstance value
    //generate a result list
    //find the neiboring cities(left and right if exists) as potential candidates
        //binary search for closest city
        //add left right return list
    //add all potential candidates together
    //set a mindistance value
    //build result list of indexes of cities based on mindistance
    //get the name of cities  from index and put into list then sort

    public static String nearestStone(String[] citieNames, int[] cityX, int[] cityY, String cityToFind){

        HashMap<Integer, List<Integer[]>> sameX = new HashMap<>();
        HashMap<Integer, List<Integer[]>> sameY = new HashMap<>();
        HashMap<String, Integer> names = new HashMap<>();
        for(int i = 0; i < cityX.length; i++){
            Integer[] c1 = new Integer[]{cityX[i], cityY[i], i};
            names.put(citieNames[i], i);
            if(!sameX.containsKey(cityX[i])){
                sameX.put(cityX[i], new ArrayList<>());
            }
            sameX.get(cityX[i]).add(c1);
            if(!sameY.containsKey(cityY[i])){
                sameY.put(cityY[i], new ArrayList<>());
            }
            sameY.get(cityY[i]).add(c1);
        }
        //sort list of cities in cityx according to y
        for(Map.Entry<Integer, List<Integer[]>> mapper : sameX.entrySet()){
            List<Integer[]> citiesInX= mapper.getValue();
            citiesInX.sort((c1,c2)->(c1[1]-c2[2]));
            mapper.setValue(citiesInX);
        }
        //sort list of cities in city according to x
         for(Map.Entry<Integer, List<Integer[]>> mapper : sameY.entrySet()){
            List<Integer[]> citiesInY= mapper.getValue();
            citiesInY.sort((c1,c2)->(c1[0]-c2[0]));
            mapper.setValue(citiesInY);
        }
        int cityIndex = names.get(cityToFind);

        //get x an y value;

        int minDistance = Integer.MAX_VALUE;
        List<Integer[]> citySameX = sameX.get(cityX[cityIndex]);
        List<Integer[]> citySameY = sameY.get(cityY[cityIndex]);

        List<Integer> XIndex = searchClosest(1, citySameX,cityY[cityIndex]);
        List<Integer> YIndex = searchClosest(0, citySameY,cityX[cityIndex]);
        XIndex.addAll(YIndex);
        List<Integer> allNeiborIndexes = new ArrayList<>();

        for(int index : XIndex){
                int distance = Math.abs(cityY[index]+cityX[index] - cityX[cityIndex]-cityY[cityIndex]);
            //  System.out.print("left");
                if( distance < minDistance){
                    minDistance = distance;
                    allNeiborIndexes = new ArrayList<>();
                    allNeiborIndexes.add(index);
                }
                else if(distance == minDistance){
                    allNeiborIndexes.add(index);
                }
        }
        if(allNeiborIndexes.isEmpty())return "";

        List<String> resultNames = new ArrayList<String>();
        for(Integer index : allNeiborIndexes){
            resultNames.add(citieNames[index]);
        }
        Collections.sort(resultNames);

        return resultNames.get(0);

    }


    private static List<Integer> searchClosest(int indexToSearch, List<Integer[]> citySameCoor, int target){

        int start = 0;
        int end = citySameCoor.size()-1;
        int targetIndex = -1;
        List<Integer> nerborIndexes = new ArrayList<>();
        while(start <= end){
            int middle = (start+end)/2;
            if(citySameCoor.get(middle)[indexToSearch] < target){
                start = middle+1;
            }
            else if(citySameCoor.get(middle)[indexToSearch] > target){
                end = middle-1;
            }
            else{
                //find target city
                targetIndex = middle;
                break;
            }
        }
        if(targetIndex-1>=0)nerborIndexes.add(citySameCoor.get(targetIndex-1)[2]);
        if(targetIndex+1<citySameCoor.size())nerborIndexes.add(citySameCoor.get(targetIndex+1)[2]);
        return nerborIndexes;

    }






    public static void main(String[] args) {
      String[] citieNames = new String[]{"c1","c2","c3","c4","c5"};
      int[] cityX = new int[]{0,0,1,1,2};
      int[] cityY = new int[]{0,1,0,1,2};
      System.out.print(nearestStone(citieNames, cityX, cityY, "c2"));
    }
}

"""

from collections import defaultdict
from typing import List


class Solution:
    from typing import List

    def nearestStone(self, cities: List[str], cityX: List[int], cityY: List[int], cityToFind: str) -> str:
        sameX = {}
        sameY = {}
        names = {}
        for i in range(len(cityX)):
            c = [cityX[i], cityY[i], i]
            names[cities[i]] = i
            if cityX[i] not in sameX:
                sameX[cityX[i]] = []
            sameX[cityX[i]].append(c)
            if cityY[i] not in sameY:
                sameY[cityY[i]] = []
            sameY[cityY[i]].append(c)
        for mapper in sameX:
            sameX[mapper].sort(key=lambda x: x[1])
        for mapper in sameY:
            sameY[mapper].sort(key=lambda x: x[0])
        cityIndex = names[cityToFind]
        minDistance = float("inf")
        citySameX = sameX[cityX[cityIndex]]
        citySameY = sameY[cityY[cityIndex]]
        XIndex = self.searchClosest(1, citySameX, cityY[cityIndex])
        YIndex = self.searchClosest(0, citySameY, cityX[cityIndex])
        allNeiborIndexes = XIndex + YIndex
        res = []
        for index in allNeiborIndexes:
            distance = abs(cityY[index] + cityX[index] - cityX[cityIndex] - cityY[cityIndex])
            if distance < minDistance:
                minDistance = distance
                res = [index]
            elif distance == minDistance:
                res.append(index)
        if not res:
            return ""
        resultNames = [cities[index] for index in res]
        return sorted(resultNames)[-1]

    def searchClosest(self, indexToSearch: int, citySameCoor: List[List[int]], target: int) -> List[int]:
        start = 0
        end = len(citySameCoor) - 1
        targetIndex = -1
        nerborIndexes = []
        while start <= end:
            middle = (start + end) // 2
            if citySameCoor[middle][indexToSearch] < target:
                start = middle + 1
            elif citySameCoor[middle][indexToSearch] > target:
                end = middle - 1
            else:
                targetIndex = middle
                break
        if targetIndex - 1 >= 0:
            nerborIndexes.append(citySameCoor[targetIndex - 1][2])
        if targetIndex + 1 < len(citySameCoor):
            nerborIndexes.append(citySameCoor[targetIndex + 1][2])
        return nerborIndexes

    def _nearestStone(self, cities: List[str], cityX: List[int], cityY: List[int], cityToFind: str) -> str:
        coords = [(cityX[i], cityY[i], cities[i]) for i in range(len(cityX))]
        target_x, target_y = coords[cities.index(cityToFind)][:2]
        neighbors = []
        for x, y, name in sorted(coords, key=lambda c: abs(c[0] - target_x) + abs(c[1] - target_y)):
            if x == target_x or y == target_y:
                neighbors.append(name)
            else:
                break
        return max(neighbors) if neighbors else ""

    def nearestStone_binary(self, cities: List[str], cityX: List[int], cityY: List[int], cityToFind: str) -> str:
        coords = [(cityX[i], cityY[i], cities[i]) for i in range(len(cityX))]
        target_x, target_y = coords[cities.index(cityToFind)][:2]
        x_coords = sorted(set(cityX))
        y_coords = sorted(set(cityY))
        x_map = defaultdict(list)
        y_map = defaultdict(list)
        for x, y, name in coords:
            x_map[x].append((y, name))
            y_map[y].append((x, name))
        x_neighbors = self.get_neighbors(x_coords, x_map, target_x, target_y, cities)
        y_neighbors = self.get_neighbors(y_coords, y_map, target_y, target_x, cities)
        neighbors = sorted(list(set(x_neighbors) | set(y_neighbors)))
        return max(cities[i] for i in neighbors) if neighbors else ""

    def get_neighbors(self, coords: List[int], coord_map: defaultdict, target_coord: int, fixed_coord: int, cities) -> \
            List[int]:
        neighbors = []
        left_idx, right_idx = 0, len(coords) - 1
        while left_idx <= right_idx:
            mid_idx = (left_idx + right_idx) // 2
            if coords[mid_idx] == target_coord:
                break
            elif coords[mid_idx] < target_coord:
                left_idx = mid_idx + 1
            else:
                right_idx = mid_idx - 1
        if coords[mid_idx] != target_coord:
            return neighbors
        for y, name in coord_map[coords[mid_idx]]:
            if y == fixed_coord:
                neighbors.append(cities.index(name))
        if mid_idx > 0:
            for y, name in coord_map[coords[mid_idx - 1]]:
                if y == fixed_coord:
                    neighbors.append(cities.index(name))
        if mid_idx < len(coords) - 1:
            for y, name in coord_map[coords[mid_idx + 1]]:
                if y == fixed_coord:
                    neighbors.append(cities.index(name))
        return neighbors


if __name__ == '__main__':

    city_names = ["c1", "c2", "c3", "c4", "c5"]
    city_x = [0, 0, 1, 1, 2]
    city_y = [0, 1, 0, 1, 2]
    from sortedcontainers import SortedList
    same_x = defaultdict(list)
    same_y = defaultdict(list)
    for index, (x, y, name) in enumerate(zip(city_x, city_y, city_names)):
        same_x[x].append([name, y])
        same_y[y].append([name, x])

    res = Solution().nearestStone_binary(cities=city_names, cityX=city_x, cityY=city_y, cityToFind="c3")
    print(res)
