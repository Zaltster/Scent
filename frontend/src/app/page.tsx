"use client"
import React, { useState } from 'react';
import { Search, Sun, Moon } from 'lucide-react';

const ScentLabHomepage = () => {
  const [searchQuery, setSearchQuery] = useState('');
  const [isDarkMode, setIsDarkMode] = useState(true);

  const handleDiscover = () => {
    if (!searchQuery.trim()) return;
    console.log('Discovering fragrance for:', searchQuery);
  };

  const toggleTheme = () => {
    setIsDarkMode(!isDarkMode);
  };

  return (
    <div
      className={`relative flex size-full min-h-screen flex-col ${isDarkMode ? 'bg-[#141122]' : 'bg-[#f9f8fc]'} ${isDarkMode ? 'dark' : ''} overflow-x-hidden`}
      style={{ fontFamily: '"Space Grotesk", "Noto Sans", sans-serif' }}
    >
      <div className="layout-container flex h-full grow flex-col">
        {/* Header */}
        <header className={`flex items-center justify-between whitespace-nowrap border-b border-solid ${isDarkMode ? 'border-b-[#2a2447]' : 'border-b-[#e9e7f3]'} px-10 py-3`}>
          <div className={`flex items-center gap-4 ${isDarkMode ? 'text-white' : 'text-[#100e1b]'}`}>
            <div className="size-4">
              <svg viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path
                  d="M36.7273 44C33.9891 44 31.6043 39.8386 30.3636 33.69C29.123 39.8386 26.7382 44 24 44C21.2618 44 18.877 39.8386 17.6364 33.69C16.3957 39.8386 14.0109 44 11.2727 44C7.25611 44 4 35.0457 4 24C4 12.9543 7.25611 4 11.2727 4C14.0109 4 16.3957 8.16144 17.6364 14.31C18.877 8.16144 21.2618 4 24 4C26.7382 4 29.123 8.16144 30.3636 14.31C31.6043 8.16144 33.9891 4 36.7273 4C40.7439 4 44 12.9543 44 24C44 35.0457 40.7439 44 36.7273 44Z"
                  fill="currentColor"
                />
              </svg>
            </div>
            <h2 className={`${isDarkMode ? 'text-white' : 'text-[#100e1b]'} text-lg font-bold leading-tight tracking-[-0.015em]`}>Odessence</h2>
          </div>
          <div className="flex flex-1 justify-end gap-8">
            <div className="flex items-center gap-9">
              <a className={`${isDarkMode ? 'text-white' : 'text-[#100e1b]'} text-sm font-medium leading-normal`} href="#">Shop</a>
              <a className={`${isDarkMode ? 'text-white' : 'text-[#100e1b]'} text-sm font-medium leading-normal`} href="#">Learn</a>
              <a className={`${isDarkMode ? 'text-white' : 'text-[#100e1b]'} text-sm font-medium leading-normal`} href="#">About</a>
            </div>
            <div className="flex gap-2">
              <button className={`flex min-w-[84px] max-w-[480px] cursor-pointer items-center justify-center overflow-hidden rounded-lg h-10 px-4 ${isDarkMode ? 'bg-[#2a2447] text-white' : 'bg-[#e9e7f3] text-[#100e1b]'} text-sm font-bold leading-normal tracking-[0.015em]`}>
                <span className="truncate">Sign In</span>
              </button>
              <button className={`flex max-w-[480px] cursor-pointer items-center justify-center overflow-hidden rounded-lg h-10 ${isDarkMode ? 'bg-[#2a2447] text-white' : 'bg-[#e9e7f3] text-[#100e1b]'} gap-2 text-sm font-bold leading-normal tracking-[0.015em] min-w-0 px-2.5`}>
                <Search className="w-5 h-5" />
              </button>
              {/* Theme Toggle Button */}
              <button
                onClick={toggleTheme}
                className={`flex cursor-pointer items-center justify-center overflow-hidden rounded-lg h-10 px-3 ${isDarkMode ? 'bg-[#2a2447] text-white hover:bg-[#3a3557]' : 'bg-[#e9e7f3] text-[#100e1b] hover:bg-[#d4d0e7]'} transition-colors duration-200`}
                title={isDarkMode ? 'Switch to light mode' : 'Switch to dark mode'}
              >
                {isDarkMode ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
              </button>
            </div>
          </div>
        </header>

        {/* Main Content */}
        <div className="px-40 flex flex-1 justify-center py-5">
          <div className="layout-content-container flex flex-col max-w-[960px] flex-1">

            {/* Hero Section */}
            <div className="@container">
              <div className="@[480px]:p-4">
                <div
                  className="flex min-h-[480px] flex-col gap-6 bg-cover bg-center bg-no-repeat @[480px]:gap-8 @[480px]:rounded-lg items-center justify-center p-4"
                  style={{
                    backgroundImage: 'linear-gradient(rgba(0, 0, 0, 0.1) 0%, rgba(0, 0, 0, 0.4) 100%), url("https://lh3.googleusercontent.com/aida-public/AB6AXuD6rlOgKugKHows_B0lyhLmarsYnmYk18UKCFr8mYj_H8uWzebCyEUR7hJ6Zl316I7svvPva4rEn7Del0MWkticU0c0csk06GxkGV4c0FsANXFFBDthwUKKKYr7Go80idPRj97nVmvjI2WLUD3ePNeSK5trQO8fdL41KBScs5SIejx4cuvggTx-8oGuWJql59_AvspWN8a_9lvsmUtjLRx1TvJs-AAk7PnkNqCABR7tAYwJafyy5FTHNFY4f1Q4WM24hM5YzCLhVaIf")'
                  }}
                >
                  <div className="flex flex-col gap-2 text-center">
                    <h1 className="text-white text-4xl font-black leading-tight tracking-[-0.033em] @[480px]:text-5xl @[480px]:font-black @[480px]:leading-tight @[480px]:tracking-[-0.033em]">
                      Discover Your Signature Scent
                    </h1>
                    <h2 className="text-white text-sm font-normal leading-normal @[480px]:text-base @[480px]:font-normal @[480px]:leading-normal">
                      Describe your ideal fragrance, and we'll reveal the science behind it.
                    </h2>
                  </div>
                  <label className="flex flex-col min-w-40 h-14 w-full max-w-[480px] @[480px]:h-16">
                    <div className="flex w-full flex-1 items-stretch rounded-lg h-full">
                      <div className={`${isDarkMode ? 'text-[#9c93c8] border-[#3c3465] bg-[#1e1a32]' : 'text-[#5a4e97] border-[#d4d0e7] bg-[#f9f8fc]'} flex border items-center justify-center pl-[15px] rounded-l-lg border-r-0`}>
                        <Search className="w-5 h-5" />
                      </div>
                      <input
                        placeholder="Describe your ideal fragrance notes"
                        className={`form-input flex w-full min-w-0 flex-1 resize-none overflow-hidden rounded-lg ${isDarkMode ? 'text-white border-[#3c3465] bg-[#1e1a32] focus:border-[#3c3465] placeholder:text-[#9c93c8]' : 'text-[#100e1b] border-[#d4d0e7] bg-[#f9f8fc] focus:border-[#d4d0e7] placeholder:text-[#5a4e97]'} focus:outline-0 focus:ring-0 border h-full px-[15px] rounded-r-none border-r-0 pr-2 rounded-l-none border-l-0 pl-2 text-sm font-normal leading-normal @[480px]:text-base @[480px]:font-normal @[480px]:leading-normal`}
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                      />
                      <div className={`flex items-center justify-center rounded-r-lg border-l-0 border ${isDarkMode ? 'border-[#3c3465] bg-[#1e1a32]' : 'border-[#d4d0e7] bg-[#f9f8fc]'} pr-[7px]`}>
                        <button
                          onClick={handleDiscover}
                          className={`flex min-w-[84px] max-w-[480px] cursor-pointer items-center justify-center overflow-hidden rounded-lg h-10 px-4 @[480px]:h-12 @[480px]:px-5 bg-[#3b19e5] ${isDarkMode ? 'text-white' : 'text-[#f9f8fc]'} text-sm font-bold leading-normal tracking-[0.015em] @[480px]:text-base @[480px]:font-bold @[480px]:leading-normal @[480px]:tracking-[0.015em]`}
                        >
                          <span className="truncate">Discover</span>
                        </button>
                      </div>
                    </div>
                  </label>
                </div>
              </div>
            </div>

            {/* How It Works Section */}
            <h2 className={`${isDarkMode ? 'text-white' : 'text-[#100e1b]'} text-[22px] font-bold leading-tight tracking-[-0.015em] px-4 pb-3 pt-5 text-center`}>How It Works</h2>
            <p className={`${isDarkMode ? 'text-white' : 'text-[#100e1b]'} text-base font-normal leading-normal pb-3 pt-1 px-4 text-center`}>
              Our unique algorithm analyzes your fragrance preferences and identifies the key chemical compounds that create your desired scent profile. Explore the science behind your perfect fragrance.
            </p>

            {/* Ingredient Cards */}
            <div className="grid grid-cols-[repeat(auto-fit,minmax(158px,1fr))] gap-3 p-4">
              <div className="flex flex-col gap-3 pb-3">
                <div
                  className="w-full bg-center bg-no-repeat aspect-square bg-cover rounded-lg"
                  style={{
                    backgroundImage: 'url("https://lh3.googleusercontent.com/aida-public/AB6AXuAjGNdceQh6Adzlr_VvsyGEDtFOSi1MXXq9Btj8muEuDddMI49ax7xHMxsL4_FfdQh9FAwfYDAKzgB3dCP8G98aJbl19zcZ845k78Svd6QuPQffLLpuLwSLwIwTagnvqFZ6xhxla35yi3Cfx-fVt0hs6DoSJHxfrRrNLrMksX7DugOUqnI_-s_sDOB7pOTvVVODzFK2ksTq6VthpttIWktxF8pg0vd58iauW8VE6C9f1uIiPLCI2Ztt5S8RMt-MJ2Jj446zyFDVmF_b")'
                  }}
                ></div>
                <div>
                  <p className={`${isDarkMode ? 'text-white' : 'text-[#100e1b]'} text-base font-medium leading-normal`}>Citrus Notes</p>
                  <p className={`${isDarkMode ? 'text-[#9c93c8]' : 'text-[#5a4e97]'} text-sm font-normal leading-normal`}>Limonene (C10H16)</p>
                </div>
              </div>
              <div className="flex flex-col gap-3 pb-3">
                <div
                  className="w-full bg-center bg-no-repeat aspect-square bg-cover rounded-lg"
                  style={{
                    backgroundImage: 'url("https://lh3.googleusercontent.com/aida-public/AB6AXuAu_bwlx9sqYdsVg_luLcwoDhtnCdJwpQApLw0Im1l_H6meQh-1lOaFsjQb6Gj0SXrnjGs6QfAUaUeOLXHZLtvTwGDlalSuja3mvOWNeexe9bBmdg6KUQ1nGY-oNUkIqrXGDqc_Wp6KzUyVfZ69PLVYqeYgizwW5qzC21P4WtDbn8osTYnZEy_XtHXVCXGp749FrKaOEKFk5RmY1CMhiFODyJ6YLGsZWbyEq4YNhOvDoNQUrFqKBV9vF6SHoazRbbUe4KR4F9pxz2bC")'
                  }}
                ></div>
                <div>
                  <p className={`${isDarkMode ? 'text-white' : 'text-[#100e1b]'} text-base font-medium leading-normal`}>Floral Undertones</p>
                  <p className={`${isDarkMode ? 'text-[#9c93c8]' : 'text-[#5a4e97]'} text-sm font-normal leading-normal`}>Linalool (C10H18O)</p>
                </div>
              </div>
              <div className="flex flex-col gap-3 pb-3">
                <div
                  className="w-full bg-center bg-no-repeat aspect-square bg-cover rounded-lg"
                  style={{
                    backgroundImage: 'url("https://lh3.googleusercontent.com/aida-public/AB6AXuA70HN65xZhPTkCrQd4SPigeXyAoFcBZq92YUuAmcFZ1kXZUqKUmh1qzvJtDvMW-1bsLkInaVEmcYSBQTOwcU1-7Huec-WQUI7ufh1By9q2t8i8MSAKZFWnVqmnM-mtx-i7rF5-9cJMGvLcTsXwUS29jZWCiq5D7-UKtNw9JlfXD3Yt6MBxu-GRYxAmRLn_D2an2biyAdORUwgkcRIHd8KgQjAI8b84cuhjq6VphJqcT9HDSis_z-4dKRXYXyWUKyoMKsAgfnaya62C")'
                  }}
                ></div>
                <div>
                  <p className={`${isDarkMode ? 'text-white' : 'text-[#100e1b]'} text-base font-medium leading-normal`}>Woody Base</p>
                  <p className={`${isDarkMode ? 'text-[#9c93c8]' : 'text-[#5a4e97]'} text-sm font-normal leading-normal`}>Cedrene (C15H24)</p>
                </div>
              </div>
            </div>

            {/* Featured Fragrances Section */}
            <h2 className={`${isDarkMode ? 'text-white' : 'text-[#100e1b]'} text-[22px] font-bold leading-tight tracking-[-0.015em] px-4 pb-3 pt-5`}>Featured Fragrances</h2>
            <div className="flex overflow-y-auto [-ms-scrollbar-style:none] [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
              <div className="flex items-stretch p-4 gap-3">
                <div className="flex h-full flex-1 flex-col gap-4 rounded-lg min-w-60">
                  <div
                    className="w-full bg-center bg-no-repeat aspect-square bg-cover rounded-lg flex flex-col"
                    style={{
                      backgroundImage: 'url("https://lh3.googleusercontent.com/aida-public/AB6AXuCo09aDi_Lt1ENAVeZsXHlyUXAs012BN_xKu51Ae78zlIzkGu2JX0_R84SSe8eJCtRYBQVxQedKs3PRaC6JB8VRtXr5QXj6luj7QGIl9RXvVRxcaZuC-pfeygMYN-BzSBw-8im6b3feswBFwfLD6w_NTBBo7ElrbWl6cpft0eWqOytNbcnQObsulPshtW44ThJx2XbD-GWeo_c1etKl6M_OXI5HChG_Id_qjl_4hmYe9UX-yjKQPpQCgkwLzmGcKCvuiTXp2jj8i7nZ")'
                    }}
                  ></div>
                  <div>
                    <p className={`${isDarkMode ? 'text-white' : 'text-[#100e1b]'} text-base font-medium leading-normal`}>Citrus Bloom</p>
                    <p className={`${isDarkMode ? 'text-[#9c93c8]' : 'text-[#5a4e97]'} text-sm font-normal leading-normal`}>A refreshing blend of citrus notes.</p>
                  </div>
                </div>
                <div className="flex h-full flex-1 flex-col gap-4 rounded-lg min-w-60">
                  <div
                    className="w-full bg-center bg-no-repeat aspect-square bg-cover rounded-lg flex flex-col"
                    style={{
                      backgroundImage: 'url("https://lh3.googleusercontent.com/aida-public/AB6AXuDJWB2Al6ONJ6gd18HSD6m2_oNn71S8kqc79e_Yd1V32bcpnEZ1z7SKtXGdw4Qx24pgkuiMDL5v-QtnU6FAZNXl0xum-dq-oJ4gxwukiFfGDYFEggwk7iF7jVMvgxxrYOcyfsQx40ImyRpnmvkMBphunTs6zQMXtPNzn7C6VBNdS9neypwVbJhylQWMd3hHKntnWawUv0cO3PiauG2ysS3WDF28LhOKYgAY46i3Dcw-Pc5pEp5pk9KVqTqQnIHmZAHcjuqPqyynyD9G")'
                    }}
                  ></div>
                  <div>
                    <p className={`${isDarkMode ? 'text-white' : 'text-[#100e1b]'} text-base font-medium leading-normal`}>Floral Essence</p>
                    <p className={`${isDarkMode ? 'text-[#9c93c8]' : 'text-[#5a4e97]'} text-sm font-normal leading-normal`}>A delicate fragrance with floral undertones.</p>
                  </div>
                </div>
                <div className="flex h-full flex-1 flex-col gap-4 rounded-lg min-w-60">
                  <div
                    className="w-full bg-center bg-no-repeat aspect-square bg-cover rounded-lg flex flex-col"
                    style={{
                      backgroundImage: 'url("https://lh3.googleusercontent.com/aida-public/AB6AXuCGZ4q7SRZYyW4mi6eq-zplZnBPqb-WILGUVqWtk5KnRC9Op2RELIyWZigd708rBQCVcgWZbayDBqbylJPdsfb_RGGiVEX_UPAGlQXW7HXQsnTS-014UUvNEeLv5sJaBeVVDlRsZa0lu3zoAwoNHOFnkyhOGh-L7H5UOcYeWG9ETZfB5ajk2rUCSWI8wP3huHuT-bMhBYVYoOJo3LK5sw8epWipuyNNa9HVh-9-Rw2hg2r4HGD5ZWkNcMoZZCJsES16ZL1lCSaE6vGo")'
                    }}
                  ></div>
                  <div>
                    <p className={`${isDarkMode ? 'text-white' : 'text-[#100e1b]'} text-base font-medium leading-normal`}>Woodland Whisper</p>
                    <p className={`${isDarkMode ? 'text-[#9c93c8]' : 'text-[#5a4e97]'} text-sm font-normal leading-normal`}>A warm and earthy scent with woody base notes.</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <footer className="flex justify-center">
          <div className="flex max-w-[960px] flex-1 flex-col">
            <footer className="flex flex-col gap-6 px-5 py-10 text-center @container">
              <div className="flex flex-wrap items-center justify-center gap-6 @[480px]:flex-row @[480px]:justify-around">
                <a className={`${isDarkMode ? 'text-[#9c93c8]' : 'text-[#5a4e97]'} text-base font-normal leading-normal min-w-40`} href="#">Privacy Policy</a>
                <a className={`${isDarkMode ? 'text-[#9c93c8]' : 'text-[#5a4e97]'} text-base font-normal leading-normal min-w-40`} href="#">Terms of Service</a>
                <a className={`${isDarkMode ? 'text-[#9c93c8]' : 'text-[#5a4e97]'} text-base font-normal leading-normal min-w-40`} href="#">Contact Us</a>
              </div>
              <p className={`${isDarkMode ? 'text-[#9c93c8]' : 'text-[#5a4e97]'} text-base font-normal leading-normal`}>@2024 Odessence. All rights reserved.</p>
            </footer>
          </div>
        </footer>
      </div>
    </div>
  );
};

export default ScentLabHomepage;